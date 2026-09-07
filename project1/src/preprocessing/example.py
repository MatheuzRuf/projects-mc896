from pathlib import Path

from .dataset import preprocess_cases


def main():
    cases_path = Path("sample/cases.csv")

    cases = preprocess_cases(cases_path)

    print(f"Casos processados: {len(cases)}")

    if not cases:
        return

    case = cases[0]

    print("\n--- ORIGINAL ---")
    print(case["original_text"][:500])

    print("\n--- NORMALIZED ---")
    print(case["normalized_text"][:500])

    print("\n--- TOKENS ---")
    print(case["tokens"][:50])

    print("\n--- RETRIEVAL TOKENS ---")
    print(case["retrieval_tokens"][:50])


if __name__ == "__main__":
    main()