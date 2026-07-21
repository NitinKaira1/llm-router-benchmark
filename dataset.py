"""
36 diverse test prompts spanning the four query types the classifier
targets. The 'expected_label' field is your ground truth for reporting
classifier accuracy separately from answer-quality accuracy -- these
are two different things and reviewers will ask about both.
"""

TEST_PROMPTS = [

    {"prompt": "What is the capital of Australia?", "expected_label": "simple"},
    {"prompt": "Convert 100 Fahrenheit to Celsius.", "expected_label": "simple"},
    {"prompt": "Who wrote 'Pride and Prejudice'?", "expected_label": "simple"},
    {"prompt": "What year did World War II end?", "expected_label": "simple"},
    {"prompt": "What is the chemical symbol for gold?", "expected_label": "simple"},
    {"prompt": "How many continents are there?", "expected_label": "simple"},
    {"prompt": "What's the plural of 'cactus'?", "expected_label": "simple"},
    {"prompt": "What is 15% of 200?", "expected_label": "simple"},
    {"prompt": "Name the largest planet in our solar system.", "expected_label": "simple"},

    {"prompt": "Write a Python function to check if a string is a palindrome.", "expected_label": "code"},
    {"prompt": "Debug this code: `def add(a, b): return a - b` -- it's supposed to add two numbers.", "expected_label": "code"},
    {"prompt": "Write a SQL query to find the second-highest salary in an Employees table.", "expected_label": "code"},
    {"prompt": "Explain what this regex does: `^(?=.*[A-Z])(?=.*\\d).{8,}$`", "expected_label": "code"},
    {"prompt": "Write a JavaScript function that debounces another function.", "expected_label": "code"},
    {"prompt": "Write a Python script to merge two sorted lists into one sorted list without using sort().", "expected_label": "code"},
    {"prompt": "Convert this for-loop into a list comprehension: `result = []\\nfor x in range(10):\\n  if x % 2 == 0:\\n    result.append(x*x)`", "expected_label": "code"},
    {"prompt": "Write a recursive function to compute the nth Fibonacci number in Python.", "expected_label": "code"},
    {"prompt": "What's wrong with this code: `for i in range(10): time.sleep(1); print(i)` if I want it to print instantly?", "expected_label": "code"},


    {"prompt": "A farmer has chickens and rabbits. Together they have 35 heads and 94 legs. How many of each animal are there?", "expected_label": "reasoning"},
    {"prompt": "If a train leaves Chicago at 9am going 60mph toward New York, and another leaves New York at 10am going 80mph toward Chicago, 900 miles apart, when do they meet?", "expected_label": "reasoning"},
    {"prompt": "You have 3 boxes: one with only apples, one with only oranges, one with both. All labels are wrong. You may pick one fruit from one box to correctly label all three. How?", "expected_label": "reasoning"},
    {"prompt": "Is the following argument valid? All mammals are warm-blooded. Whales are mammals. Therefore, whales are warm-blooded. Explain why.", "expected_label": "reasoning"},
    {"prompt": "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Explain step by step.", "expected_label": "reasoning"},
    {"prompt": "Three friends split a restaurant bill of $150 unevenly based on what they ordered, and one of them adds a 20% tip on their own share only. If person A ordered $60 worth, B ordered $50, and C ordered $40, how much does each person pay in total?", "expected_label": "reasoning"},
    {"prompt": "Explain the Monty Hall problem and why switching doors gives a 2/3 chance of winning.", "expected_label": "reasoning"},
    {"prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Explain your reasoning.", "expected_label": "reasoning"},
    {"prompt": "A staircase has 10 steps. You can climb 1 or 2 steps at a time. How many distinct ways can you climb to the top? Show the reasoning, not just the answer.", "expected_label": "reasoning"},


    {"prompt": "Write a short poem about the first snowfall of winter.", "expected_label": "creative"},
    {"prompt": "Brainstorm 5 unique names for a coffee shop that also sells houseplants.", "expected_label": "creative"},
    {"prompt": "Write a two-sentence opening line for a mystery novel set on a cruise ship.", "expected_label": "creative"},
    {"prompt": "Write a short, punchy tagline for a fitness app aimed at busy parents.", "expected_label": "creative"},
    {"prompt": "Write a 100-word fable about a fox and a river, with a moral at the end.", "expected_label": "creative"},
    {"prompt": "Come up with a creative metaphor to explain how neural networks learn, for a non-technical audience.", "expected_label": "creative"},
    {"prompt": "Write a short breakup text message from the perspective of a houseplant to its owner, humorously.", "expected_label": "creative"},
    {"prompt": "Write the opening paragraph of a fantasy story where the protagonist discovers they can talk to shadows.", "expected_label": "creative"},
    {"prompt": "Brainstorm 3 creative marketing angles for a reusable water bottle brand.", "expected_label": "creative"},
]
