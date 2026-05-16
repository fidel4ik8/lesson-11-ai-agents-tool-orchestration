# Experiment summary

```json
{
  "baseline": {
    "n_tasks": 16,
    "latency_p50_ms": 1966,
    "latency_p95_ms": 5832,
    "latency_mean_ms": 2757,
    "cost_mean_usd": 0.01019,
    "cost_total_usd": 0.163,
    "tokens_mean": 3693,
    "tokens_total": 59089,
    "score_means": {
      "escalation_correct": 1.0,
      "groundedness": 0.737,
      "judge_success": 0.688,
      "tool_selection_accuracy": 0.75
    },
    "inter_agent_overhead_pct": null,
    "cost_by_agent_total": {},
    "tokens_by_agent_total": {}
  },
  "crew": {
    "n_tasks": 16,
    "latency_p50_ms": 3973,
    "latency_p95_ms": 9290,
    "latency_mean_ms": 5037,
    "cost_mean_usd": 0.00995,
    "cost_total_usd": 0.1592,
    "tokens_mean": 3718,
    "tokens_total": 59489,
    "score_means": {
      "escalation_correct": 1.0,
      "groundedness": 0.663,
      "intent_accuracy": 0.875,
      "judge_success": 0.771,
      "tool_selection_accuracy": 0.875
    },
    "inter_agent_overhead_pct": 14.5,
    "cost_by_agent_total": {
      "router": 0.00095,
      "data_analyst": 0.13586,
      "advisor": 0.02211,
      "safety": 0.00023,
      "out_of_scope": 6e-05
    },
    "tokens_by_agent_total": {
      "router": 5009,
      "data_analyst": 48272,
      "advisor": 5172,
      "safety": 792,
      "out_of_scope": 244
    }
  },
  "by_category": {
    "advice": {
      "baseline": {
        "n_tasks": 5,
        "latency_p50_ms": 3609,
        "latency_p95_ms": 4557,
        "latency_mean_ms": 3292,
        "cost_mean_usd": 0.01136,
        "cost_total_usd": 0.0568,
        "tokens_mean": 3895,
        "tokens_total": 19477,
        "score_means": {
          "groundedness": 0.716,
          "judge_success": 0.467,
          "tool_selection_accuracy": 0.6
        },
        "inter_agent_overhead_pct": null,
        "cost_by_agent_total": {},
        "tokens_by_agent_total": {}
      },
      "crew": {
        "n_tasks": 5,
        "latency_p50_ms": 6704,
        "latency_p95_ms": 8166,
        "latency_mean_ms": 6217,
        "cost_mean_usd": 0.01494,
        "cost_total_usd": 0.0747,
        "tokens_mean": 4918,
        "tokens_total": 24592,
        "score_means": {
          "groundedness": 0.646,
          "intent_accuracy": 0.8,
          "judge_success": 0.8,
          "tool_selection_accuracy": 0.8
        },
        "inter_agent_overhead_pct": 19.2,
        "cost_by_agent_total": {
          "router": 0.00029,
          "data_analyst": 0.06034,
          "advisor": 0.01406
        },
        "tokens_by_agent_total": {
          "router": 1556,
          "data_analyst": 19951,
          "advisor": 3085
        }
      }
    },
    "fraud": {
      "baseline": {
        "n_tasks": 2,
        "latency_p50_ms": 1382,
        "latency_p95_ms": 1391,
        "latency_mean_ms": 1382,
        "cost_mean_usd": 0.00568,
        "cost_total_usd": 0.0114,
        "tokens_mean": 1977,
        "tokens_total": 3954,
        "score_means": {
          "escalation_correct": 1.0,
          "groundedness": 1.0,
          "judge_success": 1.0,
          "tool_selection_accuracy": 1.0
        },
        "inter_agent_overhead_pct": null,
        "cost_by_agent_total": {},
        "tokens_by_agent_total": {}
      },
      "crew": {
        "n_tasks": 2,
        "latency_p50_ms": 3869,
        "latency_p95_ms": 3994,
        "latency_mean_ms": 3869,
        "cost_mean_usd": 0.00017,
        "cost_total_usd": 0.0003,
        "tokens_mean": 716,
        "tokens_total": 1432,
        "score_means": {
          "escalation_correct": 1.0,
          "groundedness": 1.0,
          "intent_accuracy": 1.0,
          "judge_success": 1.0,
          "tool_selection_accuracy": 1.0
        },
        "inter_agent_overhead_pct": 35.1,
        "cost_by_agent_total": {
          "router": 0.00012,
          "safety": 0.00023
        },
        "tokens_by_agent_total": {
          "router": 640,
          "safety": 792
        }
      }
    },
    "multi_step": {
      "baseline": {
        "n_tasks": 3,
        "latency_p50_ms": 2330,
        "latency_p95_ms": 8608,
        "latency_mean_ms": 4479,
        "cost_mean_usd": 0.01163,
        "cost_total_usd": 0.0349,
        "tokens_mean": 4383,
        "tokens_total": 13149,
        "score_means": {
          "groundedness": 0.5,
          "judge_success": 0.444,
          "tool_selection_accuracy": 1.0
        },
        "inter_agent_overhead_pct": null,
        "cost_by_agent_total": {},
        "tokens_by_agent_total": {}
      },
      "crew": {
        "n_tasks": 3,
        "latency_p50_ms": 3939,
        "latency_p95_ms": 4642,
        "latency_mean_ms": 4029,
        "cost_mean_usd": 0.01211,
        "cost_total_usd": 0.0363,
        "tokens_mean": 4622,
        "tokens_total": 13868,
        "score_means": {
          "groundedness": 0.333,
          "intent_accuracy": 1.0,
          "judge_success": 0.333,
          "tool_selection_accuracy": 1.0
        },
        "inter_agent_overhead_pct": 16.4,
        "cost_by_agent_total": {
          "router": 0.00018,
          "data_analyst": 0.03038,
          "advisor": 0.00577
        },
        "tokens_by_agent_total": {
          "router": 949,
          "data_analyst": 11393,
          "advisor": 1526
        }
      }
    },
    "out_of_scope": {
      "baseline": {
        "n_tasks": 1,
        "latency_p50_ms": 1289,
        "latency_p95_ms": 1289,
        "latency_mean_ms": 1289,
        "cost_mean_usd": 0.0053,
        "cost_total_usd": 0.0053,
        "tokens_mean": 1930,
        "tokens_total": 1930,
        "score_means": {
          "groundedness": 1.0,
          "judge_success": 1.0,
          "tool_selection_accuracy": 1.0
        },
        "inter_agent_overhead_pct": null,
        "cost_by_agent_total": {},
        "tokens_by_agent_total": {}
      },
      "crew": {
        "n_tasks": 1,
        "latency_p50_ms": 12405,
        "latency_p95_ms": 12405,
        "latency_mean_ms": 12405,
        "cost_mean_usd": 0.00012,
        "cost_total_usd": 0.0001,
        "tokens_mean": 552,
        "tokens_total": 552,
        "score_means": {
          "groundedness": 1.0,
          "intent_accuracy": 1.0,
          "judge_success": 1.0,
          "tool_selection_accuracy": 1.0
        },
        "inter_agent_overhead_pct": 48.1,
        "cost_by_agent_total": {
          "router": 6e-05,
          "out_of_scope": 6e-05
        },
        "tokens_by_agent_total": {
          "router": 308,
          "out_of_scope": 244
        }
      }
    },
    "stats": {
      "baseline": {
        "n_tasks": 5,
        "latency_p50_ms": 1648,
        "latency_p95_ms": 3062,
        "latency_mean_ms": 2032,
        "cost_mean_usd": 0.01092,
        "cost_total_usd": 0.0546,
        "tokens_mean": 4115,
        "tokens_total": 20579,
        "score_means": {
          "groundedness": 0.742,
          "judge_success": 0.867,
          "tool_selection_accuracy": 0.6
        },
        "inter_agent_overhead_pct": null,
        "cost_by_agent_total": {},
        "tokens_by_agent_total": {}
      },
      "crew": {
        "n_tasks": 5,
        "latency_p50_ms": 3329,
        "latency_p95_ms": 4755,
        "latency_mean_ms": 3454,
        "cost_mean_usd": 0.00954,
        "cost_total_usd": 0.0477,
        "tokens_mean": 3809,
        "tokens_total": 19045,
        "score_means": {
          "groundedness": 0.675,
          "intent_accuracy": 0.8,
          "judge_success": 0.867,
          "tool_selection_accuracy": 0.8
        },
        "inter_agent_overhead_pct": 5.4,
        "cost_by_agent_total": {
          "router": 0.0003,
          "data_analyst": 0.04514,
          "advisor": 0.00227
        },
        "tokens_by_agent_total": {
          "router": 1556,
          "data_analyst": 16928,
          "advisor": 561
        }
      }
    }
  }
}
```
