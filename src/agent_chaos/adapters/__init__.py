# SPDX-License-Identifier: Apache-2.0
"""Optional integrations (each imports its dependency lazily):

* :mod:`agent_chaos.adapters.langgraph` - chaos node wrappers, ``heal_node`` and the
  ``SelfHealingCheckpointSaver`` checkpointer extension;
* :mod:`agent_chaos.adapters.langchain` - ``chaos_tool`` for LangChain tools;
* :mod:`agent_chaos.adapters.ma_trace` - log injections and recoveries into ma-trace episodes.
"""
