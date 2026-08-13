# Critical Issues - Quick Fix Guide

## Issue #1: Python 3.8 Compatibility - Type Hints ⚠️ CRITICAL

**Files to fix:**
- `utils/model_utils.py`
- `utils/data_utils.py`

### Quick Fix Script:

```bash
#!/bin/bash
# Fix Python 3.10+ type hints to 3.8+ compatible

# 1. Fix model_utils.py
sed -i 's/list\[str\]/List[str]/g' utils/model_utils.py
sed -i 's/dict\[str, int\]/Dict[str, int]/g' utils/model_utils.py
sed -i 's/dict\[str, float\]/Dict[str, float]/g' utils/model_utils.py

# 2. Fix data_utils.py
sed -i 's/list\[/List[/g' utils/data_utils.py
sed -i 's/dict\[/Dict[/g' utils/data_utils.py

# 3. Verify imports at top
echo "✓ Type hint fixes applied"
```

### Manual Fix:

**In `utils/model_utils.py`, line 121:**
```python
# BEFORE
def get_token_ids(
    tokenizer: Any,
    tokens: list[str]
) -> dict[str, int]:

# AFTER
def get_token_ids(
    tokenizer: Any,
    tokens: List[str]
) -> Dict[str, int]:
```

**In `utils/data_utils.py`, lines 191-196:**
```python
# BEFORE
def validate_data_structure(
    data: List[Dict],
    required_fields: List[str]
) -> Tuple[bool, List[str]]:

# AFTER (same, just verify imports)
def validate_data_structure(
    data: List[Dict],
    required_fields: List[str]
) -> Tuple[bool, List[str]]:
```

**Verify imports at top of both files:**
```python
from typing import Any, Dict, List, Optional, Tuple, Union
```

**Test:**
```bash
python -c "from utils.model_utils import get_token_ids; print('✓ OK')"
python -c "from utils.data_utils import validate_data_structure; print('✓ OK')"
```

---

## Issue #2: Import Order in data_utils.py ⚠️ CRITICAL

**File:** `utils/data_utils.py`

### Problem:
- Line 194: Uses `Tuple` type hint
- Line 376: `Tuple` is imported (TOO LATE)

### Quick Fix:

**Move line 376 imports to top:**

```python
# Current location (LINE 376):
from typing import Tuple

# MOVE TO: After line 7 (after other typing imports)
# Should be around line 9-10, right after other typing imports
```

### Manual Steps:
1. Open `utils/data_utils.py`
2. Find line 376: `from typing import Tuple`
3. Cut that line
4. Go to line 7-9 (where other `from typing import` statements are)
5. Paste before any code
6. Result should look like:
```python
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# ... module docstring ...

# All other imports, then code

def load_json(...) -> Union[Dict, List, None]:
```

**Test:**
```bash
python -c "
import sys
sys.path.insert(0, '.')
from utils.data_utils import validate_data_structure
print('✓ Tuple type hint resolved correctly')
"
```

---

## Issue #3: Unsafe Float Comparisons ⚠️ CRITICAL

**File:** `python/4_scoring_PRM_v2.py`
**Lines:** 247-248, 252, 283-284

### Changes:

**Line 247-248 (BEFORE):**
```python
sol["PRM_min_score"] = res["min_plus_prob"] if res["min_plus_prob"] is not None else float("-inf")
sol["PRM_score"] = res["final_plus_prob"] if res["final_plus_prob"] is not None else float("-inf")
sol["PRM_score_list"] = res["plus_probs"]
```

**Line 247-248 (AFTER):**
```python
sol["PRM_min_score"] = res["min_plus_prob"]  # Already None if not available
sol["PRM_score"] = res["final_plus_prob"]  # Already None if not available
sol["PRM_score_list"] = res["plus_probs"]
```

**Line 283-284 (BEFORE):**
```python
valid = [s for s in sols if s["PRM_min_score"] != float("-inf")]
prm_pred = max(valid, key=lambda s: s["PRM_min_score"]) if valid else None
```

**Line 283-284 (AFTER):**
```python
valid = [s for s in sols if s.get("PRM_min_score") is not None]
prm_pred = max(valid, key=lambda s: s["PRM_min_score"]) if valid else None
```

### Full Diff:

```diff
--- a/python/4_scoring_PRM_v2.py
+++ b/python/4_scoring_PRM_v2.py
@@ -244,9 +244,9 @@ def process_json_with_prm():
                     res = scoring_utils.get_prm_scores(
                         model,
                         tokenizer,
                         raw,
                         plus_id,
                         minus_id
                     )

-                    sol["PRM_min_score"] = res["min_plus_prob"] if res["min_plus_prob"] is not None else float("-inf")
-                    sol["PRM_score"] = res["final_plus_prob"] if res["final_plus_prob"] is not None else float("-inf")
+                    sol["PRM_min_score"] = res["min_plus_prob"]
+                    sol["PRM_score"] = res["final_plus_prob"]
                     sol["PRM_score_list"] = res["plus_probs"]

                 # PRM 기반 정답 여부
-                valid = [s for s in sols if s["PRM_min_score"] != float("-inf")]
+                valid = [s for s in sols if s.get("PRM_min_score") is not None]
                 prm_pred = max(valid, key=lambda s: s["PRM_min_score"]) if valid else None
```

---

## Issue #4: Missing Input Validation ⚠️ CRITICAL

**File:** `utils/scoring_utils.py`
**Function:** `get_prm_scores` (lines 17-50)

### Add validation at function start:

```python
def get_prm_scores(
    model: Any,
    tokenizer: Any,
    text: str,
    plus_id: int,
    minus_id: int,
    step_marker: str = STEP_MARKER,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """..."""
    if device is None:
        device = model.device

    # ✅ ADD THIS VALIDATION
    if not isinstance(plus_id, int) or not isinstance(minus_id, int):
        raise TypeError(f"Token IDs must be integers: plus_id={plus_id}, minus_id={minus_id}")

    # Encode text with offset mapping to find step positions
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offsets = encoded["offset_mapping"][0]

    # Get model logits
    with torch.no_grad():
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).logits[0]  # Shape: [seq_len, vocab_size]

    # ✅ ADD THIS BOUNDS CHECK
    vocab_size = logits.shape[-1]
    if plus_id >= vocab_size:
        raise ValueError(
            f"plus_id {plus_id} out of bounds for vocab_size {vocab_size}"
        )
    if minus_id >= vocab_size:
        raise ValueError(
            f"minus_id {minus_id} out of bounds for vocab_size {vocab_size}"
        )

    # Rest of function unchanged...
```

---

## Verification Checklist

After applying all fixes:

```bash
# 1. Test imports
python -c "from utils import model_utils; print('✓ model_utils OK')"
python -c "from utils import data_utils; print('✓ data_utils OK')"

# 2. Test scoring function
python -c "
from utils.scoring_utils import get_prm_scores
import inspect
sig = inspect.signature(get_prm_scores)
print(f'✓ get_prm_scores signature: {sig}')
"

# 3. Test with actual data (if available)
cd /c/Users/YK/med-prm-vl
python python/4_scoring_PRM_v2.py --help
echo "✓ Script loads without errors"

# 4. Quick functional test
python << 'EOF'
from utils.text_utils import format_question_with_options
item = {"question": "What is 2+2?", "options": ["3", "4", "5"]}
result = format_question_with_options(item)
assert "(A) 3" in result
print("✓ text_utils working")
EOF
```

---

## Summary

| Issue | File | Time | Status |
|-------|------|------|--------|
| #1: Type hints | model_utils.py, data_utils.py | 30 min | 📋 Ready |
| #2: Import order | data_utils.py | 15 min | 📋 Ready |
| #3: Float comparisons | 4_scoring_PRM_v2.py | 30 min | 📋 Ready |
| #4: Input validation | scoring_utils.py | 20 min | 📋 Ready |
| **Total** | - | **95 min** | **🟢 2 hour sprint** |

After fixes: **ALL CRITICAL ISSUES RESOLVED** ✅
