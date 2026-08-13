import unittest
from types import SimpleNamespace

from context_graph import ContextGraphIndex
from unified_context_graph import (
    UnifiedContextGraphIndex,
    evidence_gate_passes,
    extract_unified_facts,
)


class CodeBlock:
    def __init__(self, file_path, code_content, language="java", block_type="base"):
        self.file_path = file_path
        self.description = file_path
        self.code_content = code_content
        self.language = language
        self._type = block_type

    def __str__(self):
        return self.code_content


class Example:
    def __init__(self, task_id, file_path, left_context, language):
        self.task_id = task_id
        self.file_path = file_path
        self.left_context = left_context
        self.language = language


def graph_args(**overrides):
    values = {
        "ucm_graph_max_seed": 20,
        "ucm_graph_max_neighbors_per_seed": 2,
        "ucm_graph_max_expanded": 20,
        "ucm_graph_disable_same_file_edges": True,
        "ucm_graph_enable_identifier_edges": False,
        "ucm_graph_enable_import_edges": False,
        "ucm_graph_enable_api_call_edges": False,
        "ucm_graph_enable_typed_dependency_edges": False,
        "ucm_graph_enable_unified_context_edges": True,
        "ucm_graph_enable_multi_evidence": True,
        "ucm_graph_unified_query_context_lines": 160,
        "ucm_graph_unified_query_max": 16,
        "ucm_graph_unified_max_df": 12,
        "ucm_graph_max_evidence_per_candidate": 12,
        "ucm_graph_typed_allow_target_file": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class UnifiedContextGraphTest(unittest.TestCase):
    def test_java_receiver_and_inheritance_paths_are_combined(self):
        task_id = "task/java"
        base = CodeBlock(
            "src/BaseService.java",
            "public class BaseService {\n"
            "  public User findUser(String id) { return null; }\n"
            "}",
        )
        service = CodeBlock(
            "src/UserService.java",
            "public class UserService extends BaseService {\n"
            "  public User findUser(String id) { return super.findUser(id); }\n"
            "}",
        )
        index = UnifiedContextGraphIndex({task_id: [base, service]})

        matches, facts = index.neighbors(
            task_id,
            "UserService service;\n"
            "if (service != null) {\n"
            "  return service.findUser(id);",
            "java",
            max_results=16,
            max_df=12,
            origin="query",
        )

        by_path = {match["block"].file_path: match for match in matches}
        service_relations = set(by_path["src/UserService.java"]["relations"])
        base_relations = set(by_path["src/BaseService.java"]["relations"])
        self.assertIn("unified_receiver_member", service_relations)
        self.assertIn("unified_override", base_relations)
        self.assertGreaterEqual(len(by_path["src/UserService.java"]["evidence"]), 2)
        self.assertEqual(1, facts["receiver_calls"])
        self.assertIn("if", facts["control_kinds"])

    def test_python_incomplete_prefix_uses_regex_receiver_binding(self):
        task_id = "task/python"
        client = CodeBlock(
            "client.py",
            "class UserClient:\n"
            "    def load_user(self, user_id):\n"
            "        return None",
            language="python",
        )
        index = UnifiedContextGraphIndex({task_id: [client]})

        matches, facts = index.neighbors(
            task_id,
            "client: UserClient\nif enabled:\n    result = client.load_user(",
            "python",
            max_results=8,
            max_df=12,
            origin="query",
        )

        self.assertTrue(matches)
        self.assertIn(
            "unified_receiver_member", set(matches[0]["relations"])
        )
        self.assertEqual(1, facts["variable_type_bindings"])
        self.assertIn("if", facts["control_kinds"])

    def test_context_graph_annotates_existing_candidate_with_all_evidence(self):
        task_id = "task/integration"
        service = CodeBlock(
            "src/UserService.java",
            "public class UserService {\n"
            "  public User findUser(String id) { return null; }\n"
            "}",
        )
        index = ContextGraphIndex({task_id: [service]})
        example = Example(
            task_id,
            "src/Target.java",
            "UserService service;\nreturn service.findUser(id);",
            "java",
        )

        expanded, trace = index.expand(example, [service], graph_args())

        self.assertEqual(1, len(expanded))
        evidence = expanded[0]._ucm_graph_evidence
        relations = {item["relation"] for item in evidence}
        self.assertIn("unified_receiver_member", relations)
        self.assertIn("unified_type_definition", relations)
        self.assertGreater(expanded[0]._ucm_graph_score, 0)
        self.assertEqual(1, trace["graph_annotated_existing"])
        self.assertGreater(trace["graph_unified_path_count"], 1)

    def test_target_file_is_excluded_from_unified_expansion(self):
        task_id = "task/exclusion"
        external = CodeBlock(
            "src/External.java",
            "public class External {\n"
            "  public User loadUser() { return null; }\n"
            "}",
        )
        target = CodeBlock(
            "src/Target.java",
            "public class Target {\n"
            "  public User loadUser() { return cached; }\n"
            "}",
        )
        index = UnifiedContextGraphIndex({task_id: [external, target]})

        matches, _ = index.neighbors(
            task_id,
            "return helper.loadUser();",
            "java",
            max_results=8,
            max_df=12,
            origin="query",
            exclude_file_path="src/Target.java",
        )

        self.assertTrue(matches)
        self.assertFalse(
            any(match["block"].file_path == "src/Target.java" for match in matches)
        )

    def test_graph_only_gate_accepts_multi_path_or_high_confidence(self):
        low = (
            {"relation": "unified_call_definition", "confidence": 2.0},
        )
        high = (
            {"relation": "unified_receiver_member", "confidence": 3.2},
        )
        multi = (
            {"relation": "unified_call_definition", "confidence": 2.1},
            {"relation": "unified_type_definition", "confidence": 2.2},
        )

        self.assertFalse(evidence_gate_passes(low, 2, 3.0))
        self.assertTrue(evidence_gate_passes(high, 2, 3.0))
        self.assertTrue(evidence_gate_passes(multi, 2, 3.0))

    def test_extract_java_signature_bindings(self):
        facts = extract_unified_facts(
            "public User load(UserRequest request, Context context) { return null; }",
            "java",
            visible_prefix=True,
        )

        self.assertIn("User", facts["expected_types"])
        self.assertIn("UserRequest", facts["parameter_types"])


if __name__ == "__main__":
    unittest.main()
