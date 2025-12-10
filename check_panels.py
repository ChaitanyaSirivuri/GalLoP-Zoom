"""Quick check of associated_figure_panels column format"""
from datasets import load_dataset

ds = load_dataset('StonyBrookNLP/MuSciClaims')
df = ds['test']

# Collect all unique panel formats
all_panels = set()
for i in range(len(df)):
    panels = df[i].get('associated_figure_panels', [])
    for p in panels:
        all_panels.add(str(p))

print("All unique panel formats in dataset:")
for p in sorted(all_panels):
    print(f"  {repr(p)}")

print(f"\nTotal unique panel formats: {len(all_panels)}")

# Show examples
print("\n--- First 20 examples ---")
for i in range(20):
    row = df[i]
    filename = row['associated_figure_filepath'].split('/')[-1]
    panels = row.get('associated_figure_panels', [])
    print(f"[{i}] {filename}")
    print(f"     Panels: {panels}")
