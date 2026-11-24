# Third-Party Sources

The files below are copied unchanged from [aiming-lab/SkillRL](https://github.com/aiming-lab/SkillRL/tree/4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d), commit `4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d`.

| Local File | Upstream Path |
| --- | --- |
| `skillem/vendor/search_prompts.py` | `agent_system/environments/prompts/search.py` |
| `skillem/vendor/search_projection.py` | `agent_system/environments/env_package/search/projection.py` |
| `skillem/vendor/search_metrics.py` | `agent_system/environments/env_package/search/third_party/skyrl_gym/envs/search/utils.py` |

The copied files carry Apache-2.0 copyright notices. The Apache-2.0 license is included at `skillem/vendor/LICENSE.Apache-2.0`. SkillRL's repository-level MIT license is preserved separately at `skillem/vendor/LICENSE.SkillRL`. The metrics implementation credits Search-R1. The downloaded source archive retains its own licenses.

`skillem/bank.py` follows the schema compatibility approach in the local SearchLM-ExSkill fork and implements a separate lexical baseline. The upstream SkillRL runtime, the lexical baseline, and generated SkillEM skills have distinct retrieval behavior.

Downloaded datasets, model weights, and Wikipedia assets retain their original terms. This project records their sources and revisions; it does not assign a new license to them.

`data/search_skills_schema_corrected.json` derives from the pinned upstream `memory_data/search/claude_style_skills_search.json`. It adds the `task_specific_skills` alias without changing skill contents. The original bank remains in the downloaded source.
