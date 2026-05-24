"""Phase 1 verification: ConversationMemory evicts old messages after max_turns."""

import sys
import os

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agents"))

from agents.memory import ConversationMemory

# --- Setup ---
mem = ConversationMemory(max_turns=10)

# Anchor turn pair (turn 1)
mem.add("user", "What is the maximum daily dose of paracetamol for a healthy adult?")
mem.add("assistant", "The maximum daily dose of paracetamol for a healthy adult is 4000 mg (4 g) per day.")

# 11 distractor turn pairs (turns 2-12)
distractors = [
    ("How much sleep do adults need?", "Adults need 7-9 hours of sleep per night."),
    ("What is the normal resting heart rate?", "A normal resting heart rate for adults is 60-100 beats per minute."),
    ("How much water should I drink daily?", "Adults should drink about 2-3 litres of water per day."),
    ("What is a healthy BMI range?", "A healthy BMI is generally considered to be between 18.5 and 24.9."),
    ("How often should adults exercise?", "Adults should aim for at least 150 minutes of moderate exercise per week."),
    ("What is the normal blood pressure?", "Normal blood pressure is below 120/80 mmHg."),
    ("What is the average life expectancy?", "The global average life expectancy is approximately 73 years."),
    ("How many calories does an adult need daily?", "An average adult needs approximately 2000-2500 calories per day."),
    ("What is the normal fasting blood sugar?", "Normal fasting blood sugar is 70-99 mg/dL."),
    ("How long does the flu last?", "The flu typically lasts 5-7 days."),
    ("What is a normal body temperature?", "Normal body temperature is around 37°C (98.6°F)."),
]

for user_msg, assistant_msg in distractors:
    mem.add("user", user_msg)
    mem.add("assistant", assistant_msg)

# --- Verification ---
messages = mem.messages()

# Check for anchor presence
anchor_present = any(
    "4000" in msg.get("content", "")
    or "paracetamol max" in msg.get("content", "")
    or "4 g per day" in msg.get("content", "")
    for msg in messages
)

# Check buffer length
buffer_len = len(mem)

if not anchor_present and buffer_len == 20:
    print(f"Phase 1 PASSED — anchor evicted after 11 distractor turns. Buffer size = {buffer_len} messages.")
elif anchor_present:
    print("Phase 1 FAILED — anchor still present after 11 turns!")
    sys.exit(1)
else:
    print(f"Phase 1 FAILED — unexpected buffer size: {buffer_len} (expected 20).")
    sys.exit(1)
