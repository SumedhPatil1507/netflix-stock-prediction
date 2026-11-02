"""
Simple test script to verify narrator components work without requiring API keys.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

def test_basic_imports():
    """Test that all narrator modules can be imported."""
    print("Testing basic imports...")
    try:
        from src.narrator.vector_store import VectorStore
        from src.narrator.corpus import CorpusManager
        print("[OK] VectorStore and CorpusManager imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_vector_store():
    """Test vector store initialization."""
    print("\nTesting vector store initialization...")
    try:
        from src.narrator.vector_store import VectorStore
        vs = VectorStore()
        stats = vs.get_collection_stats()
        print(f"[OK] Vector store initialized: {stats}")
        return True
    except Exception as e:
        print(f"[FAIL] Vector store test failed: {e}")
        return False

def test_corpus_manager():
    """Test corpus manager initialization."""
    print("\nTesting corpus manager...")
    try:
        from src.narrator.corpus import CorpusManager
        cm = CorpusManager()
        docs, metadatas, ids = cm.prepare_documents_for_vector_store("NFLX")
        print(f"[OK] Corpus manager prepared {len(docs)} documents")
        return True
    except Exception as e:
        print(f"[FAIL] Corpus manager test failed: {e}")
        return False

def test_agent_traces():
    """Test agent traces initialization."""
    print("\nTesting agent traces...")
    try:
        from src.agent_traces import AgentTracer
        tracer = AgentTracer(enable_tracing=False)  # Test without actual credentials
        print("[OK] Agent tracer initialized (tracing disabled)")
        return True
    except Exception as e:
        print(f"[FAIL] Agent traces test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation module."""
    print("\nTesting evaluation module...")
    try:
        from src.narrator.eval import NarrativeEvaluator
        evaluator = NarrativeEvaluator(enable_evaluation=False)  # Test without RAGAS
        print("[OK] Narrative evaluator initialized (evaluation disabled)")
        return True
    except Exception as e:
        print(f"[FAIL] Evaluation test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("AI Narrator Component Tests")
    print("=" * 60)
    
    results = []
    results.append(("Basic Imports", test_basic_imports()))
    results.append(("Vector Store", test_vector_store()))
    results.append(("Corpus Manager", test_corpus_manager()))
    results.append(("Agent Traces", test_agent_traces()))
    results.append(("Evaluation", test_evaluation()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))
    sys.exit(0 if all_passed else 1)