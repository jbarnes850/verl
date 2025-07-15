Core Rules
No Fallbacks for Non-Working Implementations: If the actual implementation does not work as intended, do not introduce fallbacks or workarounds. Stick strictly to the specified approach and report issues for clarification if needed.

Clean Code with Purposeful Comments: Maintain clean, readable code. Do not add comments explaining what changes were made; instead, comments should describe the intended purpose and functionality of the code sections.

Avoid Extra File Creation: Do not create new files if an existing file can be modified to achieve the same task. Prioritize editing and reusing current resources.
Acquire Knowledge of Implementation First: Before implementing any changes, thoroughly understand the overall system architecture, dependencies, and existing implementation details through analysis or queries.

Review Current File Knowledge: Always examine and comprehend the contents, structure, and purpose of the current file before making any modifications to it.
Justify Every Change: Be fully aware of the reason for each change. No code alterations should be made without complete certainty of their necessity and impact; justify changes based on facts, not assumptions.

No Hallucinations Allowed: Avoid fabricating information or code. If uncertain about any aspect of the task, ask for clarification to resolve confusion rather than using trial-and-error methods.

Additional Best Practices Rules

Adhere to Coding Standards: Follow established coding conventions for the language in use (e.g., PEP 8 for Python, Google Java Style for Java). Ensure consistency in naming, indentation, and structure to promote readability.

Prioritize Readability and Maintainability: Write code that is easy to understand and modify. Use meaningful variable names, break down complex logic into functions, and avoid unnecessary complexity.

Incorporate Error Handling: Always include appropriate error handling and validation to make the code robust, but only where it aligns with the task requirements without adding unrequested features.

Test Changes Incrementally: After making changes, mentally simulate or describe tests to verify functionality. If possible, suggest or outline unit tests to confirm the code works as expected.

Document Assumptions: Clearly state any assumptions made about the environment, inputs, or dependencies in the code comments or response, ensuring transparency.

Optimize for Efficiency: Consider time and space complexity when implementing solutions. Avoid inefficient algorithms unless specified, and explain trade-offs if relevant.

Use Version Control Mindset: Treat changes as if committing to a repository—ensure they are atomic, reversible, and well-reasoned, even if no actual VCS is involved.

Seek Minimal Viable Changes: Make the smallest set of changes necessary to achieve the goal, reducing the risk of introducing bugs or side effects.