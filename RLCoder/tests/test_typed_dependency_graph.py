import unittest
from types import SimpleNamespace

from context_graph import ContextGraphIndex
from typed_dependency_graph import TypedDependencyIndex, extract_dependency_facts


class CodeBlock:
    def __init__(self, file_path, description, code_content, language, block_type):
        self.file_path = file_path
        self.description = description
        self.code_content = code_content
        self.language = language
        self._type = block_type


class Example:
    def __init__(
        self,
        task_id,
        file_path,
        left_context,
        right_context,
        related_files,
        target_code,
        language,
    ):
        self.task_id = task_id
        self.file_path = file_path
        self.left_context = left_context
        self.right_context = right_context
        self.related_files = related_files
        self.target_code = target_code
        self.language = language


class TypedDependencyGraphTest(unittest.TestCase):
    def test_extracts_java_definitions_calls_and_types(self):
        definition_facts = extract_dependency_facts(
            "public class UserService {\n"
            "  public User findUser(String id) { return null; }\n"
            "}",
            "java",
        )
        query_facts = extract_dependency_facts(
            "User user = service.findUser(id);", "java"
        )

        self.assertIn("UserService", definition_facts["definitions"])
        self.assertIn("findUser", definition_facts["definitions"])
        self.assertIn("findUser", query_facts["calls"])
        self.assertIn("User", query_facts["types"])

    def test_python_regex_fallback_handles_incomplete_query(self):
        facts = extract_dependency_facts(
            "result = repository.load_user(user_id)\nif result", "python"
        )

        self.assertIn("load_user", facts["calls"])
        self.assertIn("repository", facts["uses"])

    def test_query_dependency_retrieves_definition_and_excludes_target_file(self):
        task_id = "task/1"
        blocks = {
            task_id: [
                CodeBlock(
                    "src/UserService.java",
                    "service",
                    "public class UserService {\n"
                    "  public User findUser(String id) { return null; }\n"
                    "}",
                    "java",
                    "",
                ),
                CodeBlock(
                    "src/Target.java",
                    "target",
                    "public User findUser(String id) { return cached; }",
                    "java",
                    "",
                ),
            ]
        }
        index = TypedDependencyIndex(blocks)
        matches = index.neighbors(
            task_id,
            "User user = service.findUser(id);",
            "java",
            max_results=8,
            max_df=12,
            weights={
                "graph_typed_call": 1.8,
                "graph_typed_type": 2.0,
                "graph_typed_def_use": 1.4,
            },
            query_bonus=1.0,
            origin="query",
            exclude_file_path="src/Target.java",
        )

        self.assertTrue(matches)
        self.assertTrue(
            any(
                match["relation"] == "graph_typed_call"
                and "findUser" in match["symbols"]
                and match["block"].file_path == "src/UserService.java"
                for match in matches
            )
        )
        self.assertFalse(
            any(match["block"].file_path == "src/Target.java" for match in matches)
        )

    def test_relation_allowlist_filters_before_ranking(self):
        task_id = "task/relation-filter"
        definition = CodeBlock(
            "src/UserService.java",
            "service",
            "public class UserService {\n"
            "  public User findUser(String id) { return null; }\n"
            "}",
            "java",
            "",
        )
        index = TypedDependencyIndex({task_id: [definition]})
        common_kwargs = {
            "max_results": 8,
            "max_df": 12,
            "weights": {
                "graph_typed_call": 1.8,
                "graph_typed_type": 2.0,
                "graph_typed_def_use": 1.4,
            },
            "query_bonus": 1.0,
            "origin": "query",
        }

        type_matches = index.neighbors(
            task_id,
            "UserService service = provider.findUser(id);",
            "java",
            allowed_relations={"graph_typed_type"},
            **common_kwargs,
        )
        call_matches = index.neighbors(
            task_id,
            "UserService service = provider.findUser(id);",
            "java",
            allowed_relations={"graph_typed_call"},
            **common_kwargs,
        )

        self.assertEqual(["graph_typed_type"], [m["relation"] for m in type_matches])
        self.assertEqual(["graph_typed_call"], [m["relation"] for m in call_matches])

    def test_context_graph_records_typed_metadata(self):
        task_id = "task/2"
        definition = CodeBlock(
            "src/Service.java",
            "service",
            "public class Service {\n"
            "  public User loadUser(String id) { return null; }\n"
            "}",
            "java",
            "",
        )
        index = ContextGraphIndex({task_id: [definition]})
        example = Example(
            task_id,
            "src/Target.java",
            "User user = service.loadUser(id);",
            "",
            [],
            "",
            "java",
        )
        args = SimpleNamespace(
            ucm_graph_max_seed=20,
            ucm_graph_max_neighbors_per_seed=2,
            ucm_graph_max_expanded=20,
            ucm_graph_disable_same_file_edges=True,
            ucm_graph_enable_identifier_edges=False,
            ucm_graph_enable_import_edges=False,
            ucm_graph_enable_api_call_edges=False,
            ucm_graph_enable_typed_dependency_edges=True,
            ucm_graph_typed_relation_mode="call_only",
            ucm_graph_typed_query_context_lines=80,
            ucm_graph_typed_query_max=8,
            ucm_graph_typed_max_df=12,
            ucm_graph_typed_query_bonus=1.0,
            ucm_graph_typed_call_weight=1.8,
            ucm_graph_typed_type_weight=2.0,
            ucm_graph_typed_def_use_weight=1.4,
            ucm_graph_typed_allow_target_file=False,
        )

        expanded, trace = index.expand(example, [], args)

        self.assertEqual(1, len(expanded))
        self.assertEqual("graph_typed_call", expanded[0]._ucm_graph_relation)
        self.assertIn("loadUser", expanded[0]._ucm_graph_matched_symbols)
        self.assertEqual({"query": 1}, trace["graph_typed_origins"])
        self.assertEqual("call_only", trace["graph_typed_relation_mode"])


if __name__ == "__main__":
    unittest.main()
