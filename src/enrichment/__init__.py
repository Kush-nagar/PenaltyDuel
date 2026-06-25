"""Player attribute enrichment (FBref + Transfermarkt) and entity resolution.

This package adds external player attributes (preferred foot, DOB, height) and
FBref career penalty counts to the StatsBomb penalty dataset.  It never mutates
the StatsBomb outputs in place — it only produces new files under
``outputs/enrichment/``.
"""
