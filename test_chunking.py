from pprint import pprint

from backend.indexer.cloner import get_python_files
from backend.indexer.chunker import (
    chunk_python_file,
    chunk_repo,
)


repo_path = "./tmp/repos/30a2f204"


def main():

    print("\n==============================")
    print("CODELENS AST CHUNKING TEST")
    print("==============================\n")

    python_files = get_python_files(
        local_path=repo_path,
        max_files=200,
    )

    print(f"[INFO] Python files found: {len(python_files)}")

    if not python_files:
        print("[ERROR] No Python files found.")
        return

    test_file = python_files[0]

    print("\n==============================")
    print("TESTING SINGLE FILE")
    print("==============================\n")

    print(f"[FILE] {test_file}\n")

    file_chunks = chunk_python_file(
        absolute_path=test_file,
        local_repo_path=repo_path,
    )

    print(f"[INFO] Chunks extracted: {len(file_chunks)}")

    if not file_chunks:
        print("[ERROR] No chunks extracted.")
        return

    for i, chunk in enumerate(file_chunks[:2], start=1):

        print("\n------------------------------")
        print(f"CHUNK #{i}")
        print("------------------------------")

        pprint(chunk["metadata"])

        print("\nCODE PREVIEW:\n")

        print(chunk["text"][:400])

        print("\n==============================")

    print("\n==============================")
    print("TESTING WHOLE REPOSITORY")
    print("==============================\n")

    all_chunks = chunk_repo(
        python_files=python_files,
        local_repo_path=repo_path,
    )

    print(f"[INFO] Total chunks created: {len(all_chunks)}")

    function_chunks = sum(
        1
        for c in all_chunks
        if c["metadata"]["chunk_type"] == "function"
    )

    class_chunks = sum(
        1
        for c in all_chunks
        if c["metadata"]["chunk_type"] == "class"
    )

    module_chunks = sum(
        1
        for c in all_chunks
        if c["metadata"]["chunk_type"] == "module"
    )

    print("\n==============================")
    print("CHUNK STATISTICS")
    print("==============================\n")

    print(f"Functions : {function_chunks}")
    print(f"Classes   : {class_chunks}")
    print(f"Modules   : {module_chunks}")

    print("\n[SUCCESS] AST chunking pipeline operational.\n")


if __name__ == "__main__":
    main()
