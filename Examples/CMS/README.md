# Conference Management System

## Example analyses

| Property     | Bound | Models | Result   |
|--------------|-------|--------|----------|
| `cms_gni_3x2.hq` | `< 7` | `cms_any_paper_3x2.smv` | UNSAT |
| `cms_gni_3x2.hq` | `>= 7` | `cms_any_paper_3x2.smv` | SAT |
| `cms_gni_2x2.hq` | `>= 1` | `cms_same_paper_2x2.smv` | UNSAT |
| `equivalence_2x2.hq` | `< 4` | `cms_any_paper_2x2.smv` `cms_same_paper_2x2.smv` | UNSAT |
| `equivalence_2x2.hq` | `>= 4` | `cms_any_paper_2x2.smv` `cms_same_paper_2x2.smv` | SAT |
| `equivalence_2x2.hq` | `>= 1` | `cms_same_paper_2x2.smv` `cms_any_paper_2x2.smv` | UNSAT |