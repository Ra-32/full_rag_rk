from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

# ============================================================
# SAMPLE TEXT
# ============================================================

text = """
The police investigated a robbery that happened at a local jewelry store on Saturday night. After checking the security cameras, they identified a man named John. John was arrested the next morning after the police found several pieces of stolen jewelry inside his house.

The investigators later discovered that John had entered the store shortly before closing time. He had previously worked at the same store and knew where the most expensive jewelry was kept. The police also found fingerprints near the display cabinets that matched John's fingerprints.

During the investigation, the police interviewed several witnesses who had seen John near the store on the night of the robbery. One witness reported seeing John carrying a black bag when he left the building. The investigators also recovered messages between John and another person discussing the stolen jewelry.

The police later identified the second person as Michael. Michael admitted that he had helped John hide the stolen jewelry after the robbery. The police recovered the remaining jewelry from a storage room and returned it to the store owner after completing the investigation.
"""


# ============================================================
# 1. FIXED-SIZE CHUNKING
# ============================================================

fixed = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0,
    length_function=len
)

print("=" * 70)
print("1. FIXED-SIZE CHUNKING")
print("=" * 70)

fixed_chunks = fixed.split_text(text)

print(f"Total chunks: {len(fixed_chunks)}")

for i, chunk in enumerate(fixed_chunks, 1):
    print(f"\nChunk {i}")
    print("-" * 50)
    print(chunk)
    print(f"\nCharacters: {len(chunk)}")


# ============================================================
# 2. PARAGRAPH-BASED CHUNKING
# ============================================================

paragraph = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0,
    length_function=len
)

print("\n\n")
print("=" * 70)
print("2. PARAGRAPH-BASED CHUNKING")
print("=" * 70)

paragraph_chunks = paragraph.split_text(text)

print(f"Total chunks: {len(paragraph_chunks)}")

for i, chunk in enumerate(paragraph_chunks, 1):
    print(f"\nChunk {i}")
    print("-" * 50)
    print(chunk)
    print(f"\nCharacters: {len(chunk)}")


# ============================================================
# 3. RECURSIVE CHUNKING
# ============================================================

recursive = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

print("\n\n")
print("=" * 70)
print("3. RECURSIVE CHUNKING")
print("=" * 70)

recursive_chunks = recursive.split_text(text)

print(f"Total chunks: {len(recursive_chunks)}")

for i, chunk in enumerate(recursive_chunks, 1):
    print(f"\nChunk {i}")
    print("-" * 50)
    print(chunk)
    print(f"\nCharacters: {len(chunk)}")


# ============================================================
# SUMMARY
# ============================================================

print("\n\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Fixed-size chunks      : {len(fixed_chunks)}")
print(f"Paragraph chunks       : {len(paragraph_chunks)}")
print(f"Recursive chunks       : {len(recursive_chunks)}")