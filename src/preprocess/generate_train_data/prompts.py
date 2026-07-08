"""
Prompt templates for training data generation in the SEA-SQL pipeline.

Each module-level constant is a ``str.format``-style template whose
placeholders are filled in at call sites throughout the
``generate_train_data`` package.
"""

# ---------------------------------------------------------------------------
# Data-generation prompts
# Produce new (question, SQL) pairs that are stylistically similar to an
# existing example.
# ---------------------------------------------------------------------------

generate_question_answer_without_knowledge: str = """### Generate a similar question and the corresponding gold SQL based on the given question and gold SQL.
Your output format should include "New Question" and "New Gold SQL"
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### Gold SQL: {sql}"""

generate_question_answer_with_knowledge: str = """### Generate a similar question, knowledge and the corresponding gold SQL based on the given question, external knowledge and gold SQL.
Your output format should include "New Question", "New External Knowledge" and "New Gold SQL"
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
### Gold SQL: {sql}"""

# ---------------------------------------------------------------------------
# Self-correction prompts
# Ask the model to judge a generated SQL against its execution result and
# emit a corrected query when necessary.
# ---------------------------------------------------------------------------

prompt_with_knowledge_result: str = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
This represents the SQLite SQL query that has been generated in response to the given question, along with the resulting outcome after executing the query.
Please judge its correctness based on the execution result and the explanation for the question.
If it's incorrect, output the correct sqlite SQL query; otherwise, output the original sqlite SQL query.

### Input:
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
### SQLite SQL query: {SQL}
### Run results: {result}

### Response:
"""

prompt_without_knowledge_result: str = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
This represents the SQLite SQL query that has been generated in response to the given question, along with the resulting outcome after executing the query.
Please judge its correctness based on the execution result and the explanation for the question.
If it's incorrect, output the correct sqlite SQL query; otherwise, output the original sqlite SQL query.

### Input:
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### SQLite SQL query: {SQL}
### Run results: {result}

### Response:
"""

# ---------------------------------------------------------------------------
# Zero-shot SQL generation prompts
# Direct the model to complete a SELECT query given a schema and question.
# ---------------------------------------------------------------------------

sql_simple_prompt_kg: str = """### Complete sqlite SQL query only and with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
SELECT"""

sql_simple_prompt: str = """### Complete sqlite SQL query only and with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
SELECT"""
