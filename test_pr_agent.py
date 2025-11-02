#!/usr/bin/env python3

from app.automation.pr_agent_adapter import generate_pr_template_from_diff, generate_issue_template_from_diff

# Test PR generation
test_diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,5 @@
 def hello():
-    print('Hello')
+    print('Hello World')
+    print('This is a test')
"""

print("=== PR Generation Test ===")
pr_result = generate_pr_template_from_diff(test_diff, 'Add enhanced hello function')
if pr_result:
    print(f"Title: {pr_result['title']}")
    print(f"Body preview: {pr_result['body'][:300]}...")
else:
    print("PR generation failed")

print("\n=== Issue Generation Test ===")
issue_result = generate_issue_template_from_diff(test_diff, ['enhancement', 'chore'])
if issue_result:
    print(f"Title: {issue_result['title']}")
    print(f"Body preview: {issue_result['body'][:300]}...")
else:
    print("Issue generation failed")


