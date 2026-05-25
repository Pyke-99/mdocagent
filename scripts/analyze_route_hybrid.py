#!/usr/bin/env python3
"""
Analyze route_hybrid routing results and collect statistics.

This script analyzes the route_hybrid pipeline output to provide statistics on:
- How many questions were routed to single agent (simple route)
- How many questions were routed to multi-agent system (complex route)
- Breakdown by dataset
- Success rates for different routing strategies
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


class RouteHybridAnalyzer:
    def __init__(self, results_base_dir="/root/MDocAgent/results"):
        self.results_base_dir = results_base_dir
        self.statistics = {
            "total_questions": 0,
            "simple_route": 0,
            "complex_route": 0,
            "unknown_route": 0,
            "by_dataset": defaultdict(lambda: {
                "total": 0,
                "simple": 0,
                "complex": 0,
                "unknown": 0,
                "correct": 0,
                "simple_correct": 0,
                "complex_correct": 0,
            }),
            "files_analyzed": [],
        }

    def _parse_trace_file(self, trace_file_path: str) -> Dict[str, str]:
        """
        Parse a trace file to extract routing information.
        
        Returns a mapping of question IDs to route types.
        """
        route_map = {}
        try:
            with open(trace_file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if item.get('route') == 'route_hybrid':
                        # This question was routed to hybrid system
                        # We need to check the full trace for simple/complex
                        route_map[item.get('doc_id', '') + '_' + item.get('question', '')] = 'route_hybrid'
        except Exception as e:
            print(f"Error parsing trace file {trace_file_path}: {e}")
        
        return route_map

    def _parse_analysis_file(self, analysis_file_path: str) -> List[Dict]:
        """
        Parse analysis file to extract routing and correctness information.
        """
        items = []
        try:
            with open(analysis_file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    analysis_fields = item.get('analysis_fields', {})
                    # Check for trace information in analysis fields
                    for key in analysis_fields.keys():
                        if 'trace' in key:
                            trace = analysis_fields[key]
                            if isinstance(trace, dict):
                                items.append({
                                    'doc_id': item.get('doc_id'),
                                    'question': item.get('question'),
                                    'route': trace.get('route', 'unknown'),
                                    'trace': trace,
                                })
                                break
                    # If no trace found in analysis_fields, add item with unknown route
                    if not items or items[-1].get('question') != item.get('question'):
                        items.append({
                            'doc_id': item.get('doc_id'),
                            'question': item.get('question'),
                            'route': 'unknown',
                            'trace': {},
                        })
        except Exception as e:
            print(f"Error parsing analysis file {analysis_file_path}: {e}")
        
        return items

    def _parse_results_file(self, results_file_path: str, analysis_file_path: str = None) -> List[Dict]:
        """
        Parse results file and optionally merge with analysis data.
        """
        items = []
        analysis_data = {}
        
        # Load analysis data if available
        if analysis_file_path and os.path.exists(analysis_file_path):
            analysis_items = self._parse_analysis_file(analysis_file_path)
            for item in analysis_items:
                key = (item.get('doc_id', ''), item.get('question', ''))
                analysis_data[key] = item
        
        try:
            with open(results_file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    doc_id = item.get('doc_id', '')
                    question = item.get('question', '')
                    key = (doc_id, question)
                    
                    # Check for trace in item
                    route = 'unknown'
                    trace = {}
                    
                    # First check if trace is stored directly in results
                    for key_name in item.keys():
                        if '_trace' in key_name and item[key_name]:
                            if isinstance(item[key_name], dict):
                                trace = item[key_name]
                                route = trace.get('route', 'unknown')
                                break
                    
                    # If no trace in results, check analysis data
                    if route == 'unknown' and key in analysis_data:
                        route = analysis_data[key].get('route', 'unknown')
                        trace = analysis_data[key].get('trace', {})
                    
                    # Check if answer is correct (binary correctness)
                    is_correct = item.get('binary_correctness', False)
                    
                    items.append({
                        'doc_id': doc_id,
                        'question': question,
                        'route': route,
                        'is_correct': is_correct,
                        'answer': item.get('answer', ''),
                        'trace': trace,
                    })
        except Exception as e:
            print(f"Error parsing results file {results_file_path}: {e}")
        
        return items

    def analyze_dataset(self, dataset_name: str, result_dir: Path) -> bool:
        """
        Analyze a specific result directory for a dataset.
        """
        # Look for trace files and results files
        trace_files = list(result_dir.glob("*_trace.json"))
        results_files = list(result_dir.glob("*_results.json"))
        analysis_files = list(result_dir.glob("*_analysis.json"))
        
        if not trace_files and not results_files:
            return False
        
        # Process trace files
        route_simple_count = 0
        route_complex_count = 0
        route_unknown_count = 0
        
        for trace_file in trace_files:
            print(f"Processing: {trace_file.name}")
            items = self._parse_results_file(
                str(trace_file),
                None
            )
            
            for item in items:
                route = item.get('route', 'unknown')
                self.statistics['total_questions'] += 1
                
                if route == 'simple':
                    self.statistics['simple_route'] += 1
                    route_simple_count += 1
                elif route == 'complex':
                    self.statistics['complex_route'] += 1
                    route_complex_count += 1
                else:
                    self.statistics['unknown_route'] += 1
                    route_unknown_count += 1
                
                # Update by-dataset statistics
                dataset_stats = self.statistics['by_dataset'][dataset_name]
                dataset_stats['total'] += 1
                if item.get('is_correct'):
                    dataset_stats['correct'] += 1
                    if route == 'simple':
                        dataset_stats['simple_correct'] += 1
                    elif route == 'complex':
                        dataset_stats['complex_correct'] += 1
                
                if route == 'simple':
                    dataset_stats['simple'] += 1
                elif route == 'complex':
                    dataset_stats['complex'] += 1
                else:
                    dataset_stats['unknown'] += 1
            
            self.statistics['files_analyzed'].append(str(trace_file))
        
        # Also process results files if trace files not available
        if not trace_files:
            for results_file in results_files:
                print(f"Processing: {results_file.name}")
                # Find corresponding analysis file
                analysis_file = None
                for afile in analysis_files:
                    if afile.stem == results_file.stem:
                        analysis_file = str(afile)
                        break
                
                items = self._parse_results_file(
                    str(results_file),
                    analysis_file
                )
                
                for item in items:
                    route = item.get('route', 'unknown')
                    self.statistics['total_questions'] += 1
                    
                    if route == 'simple':
                        self.statistics['simple_route'] += 1
                        route_simple_count += 1
                    elif route == 'complex':
                        self.statistics['complex_route'] += 1
                        route_complex_count += 1
                    else:
                        self.statistics['unknown_route'] += 1
                        route_unknown_count += 1
                    
                    # Update by-dataset statistics
                    dataset_stats = self.statistics['by_dataset'][dataset_name]
                    dataset_stats['total'] += 1
                    if item.get('is_correct'):
                        dataset_stats['correct'] += 1
                        if route == 'simple':
                            dataset_stats['simple_correct'] += 1
                        elif route == 'complex':
                            dataset_stats['complex_correct'] += 1
                    
                    if route == 'simple':
                        dataset_stats['simple'] += 1
                    elif route == 'complex':
                        dataset_stats['complex'] += 1
                    else:
                        dataset_stats['unknown'] += 1
                
                self.statistics['files_analyzed'].append(str(results_file))
        
        return len(self.statistics['files_analyzed']) > 0

    def analyze_all(self):
        """
        Scan the results directory and analyze all route_hybrid outputs.
        """
        results_path = Path(self.results_base_dir)
        
        if not results_path.exists():
            print(f"Results directory not found: {self.results_base_dir}")
            return
        
        print(f"Scanning results directory: {results_path}")
        
        # Look for any trace files
        trace_files = list(results_path.glob("*/*_route_hybrid_trace.json"))
        print(f"Found {len(trace_files)} trace files")
        for trace_file in trace_files:
            print(f"  - {trace_file}")
        
        # Look for results with route in directory name
        for dataset_dir in results_path.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            print(f"\nScanning dataset: {dataset_dir.name}")
            
            # Iterate through result subdirectories
            for result_dir in dataset_dir.iterdir():
                if not result_dir.is_dir():
                    continue
                
                # Check for any trace or results files
                trace_files_dir = list(result_dir.glob("*_trace.json"))
                results_files_dir = list(result_dir.glob("*_results.json"))
                
                if trace_files_dir or results_files_dir:
                    print(f"  ├─ Found {len(trace_files_dir)} trace files and {len(results_files_dir)} result files in {result_dir.name}")
                    # This looks like a route_hybrid result directory
                    self.analyze_dataset(dataset_dir.name, result_dir)

    def generate_report(self) -> str:
        """
        Generate a formatted statistics report.
        """
        report = []
        report.append("=" * 70)
        report.append("ROUTE HYBRID ROUTING STATISTICS")
        report.append("=" * 70)
        
        total = self.statistics['total_questions']
        simple = self.statistics['simple_route']
        complex = self.statistics['complex_route']
        unknown = self.statistics['unknown_route']
        
        report.append(f"\nOVERALL STATISTICS")
        report.append("-" * 70)
        report.append(f"Total Questions:        {total:6d}")
        if total > 0:
            report.append(f"Single-Agent (Simple):  {simple:6d} ({simple/total*100:.1f}%) - 单智能体")
            report.append(f"Multi-Agent (Complex):  {complex:6d} ({complex/total*100:.1f}%) - 多智能体")
            report.append(f"Unknown Route:          {unknown:6d} ({unknown/total*100:.1f}%)")
        else:
            report.append(f"Single-Agent (Simple):  {simple:6d} - 单智能体")
            report.append(f"Multi-Agent (Complex):  {complex:6d} - 多智能体")
            report.append(f"Unknown Route:          {unknown:6d}")
        
        if total > 0:
            report.append(f"\nRoute Ratio (Simple:Complex): {simple}:{complex}")
        
        # Per-dataset statistics
        if self.statistics['by_dataset']:
            report.append(f"\n\nPER-DATASET BREAKDOWN")
            report.append("-" * 70)
            report.append(f"{'Dataset':<20} {'Total':>8} {'Simple':>8} {'Complex':>8} {'Accuracy':>10}")
            report.append("-" * 70)
            
            for dataset_name in sorted(self.statistics['by_dataset'].keys()):
                stats = self.statistics['by_dataset'][dataset_name]
                total_ds = stats['total']
                simple_ds = stats['simple']
                complex_ds = stats['complex']
                correct_ds = stats['correct']
                
                if total_ds > 0:
                    accuracy = (correct_ds / total_ds) * 100
                    report.append(
                        f"{dataset_name:<20} {total_ds:>8} {simple_ds:>8} {complex_ds:>8} {accuracy:>9.1f}%"
                    )
                    
                    # Show breakdown by route
                    if simple_ds > 0:
                        simple_acc = (stats.get('simple_correct', 0) / simple_ds) * 100
                        report.append(f"  ├─ Simple:   {simple_ds:>6} correct, accuracy: {simple_acc:>5.1f}%")
                    if complex_ds > 0:
                        complex_acc = (stats.get('complex_correct', 0) / complex_ds) * 100
                        report.append(f"  └─ Complex:  {complex_ds:>6} correct, accuracy: {complex_acc:>5.1f}%")
        
        report.append(f"\n\nFILES ANALYZED: {len(self.statistics['files_analyzed'])}")
        for file_path in self.statistics['files_analyzed'][:5]:
            report.append(f"  - {file_path}")
        if len(self.statistics['files_analyzed']) > 5:
            report.append(f"  ... and {len(self.statistics['files_analyzed']) - 5} more files")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)

    def save_report(self, output_path: str):
        """
        Save the report to a file.
        """
        report = self.generate_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze route_hybrid routing results statistics"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/root/MDocAgent/results",
        help="Path to the results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="route_hybrid_statistics.txt",
        help="Output file for the report"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Analyze a specific dataset only"
    )
    
    args = parser.parse_args()
    
    analyzer = RouteHybridAnalyzer(results_base_dir=args.results_dir)
    
    if args.dataset:
        print(f"Analyzing dataset: {args.dataset}")
        dataset_path = Path(args.results_dir) / args.dataset
        if dataset_path.exists():
            for result_dir in dataset_path.iterdir():
                if result_dir.is_dir() and 'route' in result_dir.name.lower():
                    analyzer.analyze_dataset(args.dataset, result_dir)
        else:
            print(f"Dataset directory not found: {dataset_path}")
    else:
        print("Analyzing all route_hybrid results...")
        analyzer.analyze_all()
    
    # Print and save report
    report = analyzer.generate_report()
    print("\n" + report)
    
    analyzer.save_report(args.output)


if __name__ == "__main__":
    main()
