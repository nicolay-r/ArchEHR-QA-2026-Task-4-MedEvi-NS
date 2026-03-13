GROUNDING_PROMPT = """You are grounding answer sentences to a clinical note.

Goal:
    For each answer sentence, identify the clinical note sentence(s) that support it.

Input:
    Clinician-interpreted question (Clinician-Interpreted Question)
    Clinical note excerpt with numbered sentences (Clinical Note Excerpt)
    Answer text to be grounded (Reference Answer)

Expected output:
    For each answer sentence, a set of supporting evidence sentence numbers from the clinical note excerpt.

Clinical reasoning principles:

Clinical documentation follows a progression:
    background → finding → decision → procedure → outcome.

For each answer sentence:

1. Identify which clinical episode it refers to (e.g., first ERCP, repeat ERCP, specific hospital day). Only search within that episode.

2. Determine what stage of care the answer sentence represents:
   - a finding
   - a decision/recommendation
   - a performed procedure
   - a result/outcome

3. Select the sentence(s) in the clinical note that directly document that specific stage of care.

4. Prefer explicit documentation of completed events (e.g., "was performed", "showed", "was found") rather than related background or earlier mentions.

5. Do NOT cite sentences:
   - from a different episode
   - that only share keywords
   - that describe a different stage of care

6. Cite the minimal number of sentences needed.
   Only cite a sentence if removing it would make the answer unsupported.

7. Before finalizing the citation set for an answer sentence, verify that every distinct clinical fact expressed in the sentence is explicitly supported by at least one cited clinical note sentence.
   If any clinical fact is unsupported, search again and add the necessary sentence.
   Do not output this verification step.

Output format:
Output each answer sentence followed by its supporting sentence number(s) in brackets.
Example: "The patient had elevated bilirubin [3]. ERCP was performed [5]."

---

Clinician-Interpreted Question:
{clinician_question}

Clinical Note Excerpt (numbered sentences):
{clinical_note_excerpt_numbered}

Reference Answer (ground each sentence to note sentences above):
{reference_answer}

---

Output (each answer sentence followed by its supporting sentence number(s) in brackets):
"""