#!/usr/bin/env python3
"""
Performance testing script for the RAG pipeline.
Compares different configurations to show performance improvements.
"""

import time
from rag.rag_pipeline import RAGPipeline

def test_ingestion_performance(doc_link, configs):
    """
    Test ingestion performance with different configurations.
    
    Args:
        doc_link: Document link to test
        configs: List of configuration dictionaries
    """
    print("=== RAG Pipeline Performance Test ===\n")
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"Test {i+1}: {config['name']}")
        print(f"Configuration: {config}")
        
        try:
            rag = RAGPipeline()
            start_time = time.time()
            
            rag.ingest_document(
                doc_link,
                chunk_strategy=config.get('chunk_strategy', 'sentences'),
                target_chunks=config.get('target_chunks', 50),
                batch_size=config.get('batch_size', 50),
                max_workers=config.get('max_workers', 4)
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            num_chunks = len(rag.chunks)
            
            result = {
                'config': config,
                'total_time': total_time,
                'num_chunks': num_chunks,
                'time_per_chunk': total_time / num_chunks if num_chunks > 0 else 0
            }
            
            results.append(result)
            
            print(f"✅ Completed in {total_time:.2f}s")
            print(f"   Chunks: {num_chunks}")
            print(f"   Time per chunk: {result['time_per_chunk']:.3f}s\n")
            
        except Exception as e:
            print(f"❌ Failed: {e}\n")
            results.append({
                'config': config,
                'error': str(e)
            })
    
    # Print summary
    print("=== Performance Summary ===")
    successful_results = [r for r in results if 'error' not in r]
    
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['total_time'])
        slowest = max(successful_results, key=lambda x: x['total_time'])
        
        print(f"Fastest: {fastest['config']['name']} - {fastest['total_time']:.2f}s")
        print(f"Slowest: {slowest['config']['name']} - {slowest['total_time']:.2f}s")
        
        if len(successful_results) > 1:
            improvement = (slowest['total_time'] - fastest['total_time']) / slowest['total_time'] * 100
            print(f"Improvement: {improvement:.1f}% faster")
    
    return results

if __name__ == "__main__":
    # Test configurations
    configs = [
        {
            'name': 'Optimized (Parallel + Batching)',
            'chunk_strategy': 'sentences',
            'target_chunks': 50,
            'batch_size': 50,
            'max_workers': 4
        },
        {
            'name': 'High Parallelism',
            'chunk_strategy': 'sentences',
            'target_chunks': 50,
            'batch_size': 25,
            'max_workers': 8
        },
        {
            'name': 'Large Batches',
            'chunk_strategy': 'sentences',
            'target_chunks': 50,
            'batch_size': 100,
            'max_workers': 2
        },
        {
            'name': 'Fewer Chunks',
            'chunk_strategy': 'paragraphs',
            'target_chunks': 25,
            'batch_size': 50,
            'max_workers': 4
        }
    ]
    
    # Get document link from user
    doc_link = input("Enter document link (local path or URL): ").strip()
    
    if not doc_link:
        print("No document link provided. Exiting.")
        exit(1)
    
    # Run performance test
    results = test_ingestion_performance(doc_link, configs)
    
    print("\n=== Recommendations ===")
    print("1. Use 'sentences' strategy for better semantic coherence")
    print("2. Adjust batch_size based on your API rate limits")
    print("3. Increase max_workers for faster processing (if API allows)")
    print("4. Use target_chunks to control the number of chunks")
    print("5. Monitor API rate limits and adjust accordingly") 