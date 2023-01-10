import argparse
import json
import yaml
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="This script generates a merged evaluation report from all evaluation results in a directory.")
    ap.add_argument("eval_results_dir",
                    help="Path to the directory that contains the evaluation results",
                    default="data/eval_results",
                    type=str)

    opts = ap.parse_args()

    eval_results_dir = Path(opts.eval_results_dir)
    assert eval_results_dir.exists(), f"Cannot read evaluation results at {opts.eval_results_dir}"

    eval_score_files = [Path(es) for es in glob(str(eval_results_dir.joinpath("**/scores.json")),
                                                recursive=True)]
    run_cfgs_files = [Path(cfg) for cfg in glob(str(eval_results_dir.joinpath("**/*eval_run_config*.yaml")),
                                                recursive=True)]
    psg_cfgs_files = [Path(cfg) for cfg in glob(str(eval_results_dir.joinpath("**/psg_params*.yaml")),
                                                recursive=True)]

    eval_scores: Dict[str, Dict[str, float]] = {}
    for esf in eval_score_files:
        with open(str(esf), "r") as f:
            eval_scores[str(esf.parent)] = json.load(f)

    run_cfgs: Dict[str, Dict[str, float]] = {}
    for cfg in run_cfgs_files:
        run_cfgs[str(cfg.parent)] = yaml.safe_load(cfg.read_text())

    psg_cfgs: Dict[str, Dict[str, float]] = {}
    for cfg in psg_cfgs_files:
        psg_cfgs[str(cfg.parent)] = yaml.safe_load(cfg.read_text())

    # sort scores by hits@1
    eval_scores = {run_name: scores for run_name, scores in sorted(
        eval_scores.items(), key=lambda es: es[1]["hits@1"], reverse=True)}

    best_f1_run, best_f1_scores = next(iter(eval_scores.items()))

    report = f"""
# Ontology Alignment Evaluation Report

## Best Run:
## {best_f1_run}
```json
{json.dumps(best_f1_scores, indent=2, sort_keys=False)}
```

<details>
<summary>Evaluation Run Configuration</summary>

```yaml
{yaml.dump(run_cfgs[best_f1_run], indent=2, sort_keys=False)}
```

</details>

<details>
<summary>PseudoSentenceGenerator Configuration</summary>

```yaml
{yaml.dump(psg_cfgs[best_f1_run], indent=2, sort_keys=False) }
```

</details>

### Runs Sorted By F1
    """

    for run, scores in eval_scores.items():
        report += f"""
**_{run}_**

```json
{json.dumps(scores, indent=2, sort_keys=False)}
```

<details>
<summary>Evaluation Run Configuration</summary>

```yaml
{yaml.dump(run_cfgs[run], indent=2, sort_keys=False)}
```

</details>

<details>
<summary>PseudoSentenceGenerator Configuration</summary>

```yaml
{yaml.dump(psg_cfgs[run], indent=2, sort_keys=False) }
```

</details>
"""

    time_str = str(datetime.now()).replace(" ", "_")
    fn = str(eval_results_dir.joinpath(f"evaluation_report_{time_str}.md"))
    with open(fn, "w") as f:
        f.write(report)

    print(f"Generated Merged Evaluation Report at: {fn}")
