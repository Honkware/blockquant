"""Tests for the shared EXL3 card renderer."""
import re
import sys
import types

from blockquant import cards


def _fake_hf(monkeypatch, **attrs):
    mod = types.ModuleType("huggingface_hub")
    for name, value in attrs.items():
        setattr(mod, name, value)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return mod


def test_create_model_collection_returns_slug(monkeypatch):
    class Coll:
        slug = "owner/foo-exl3-abc123"

    _fake_hf(monkeypatch, create_collection=lambda **kw: Coll())
    slug = cards.create_model_collection(owner="owner", base_name="Foo-7B", token="t")
    assert slug == "owner/foo-exl3-abc123"


def test_add_to_collection_is_called(monkeypatch):
    calls = []
    _fake_hf(monkeypatch, add_collection_item=lambda **kw: calls.append(kw))
    cards.add_to_collection("owner/foo-abc", "owner/Foo-7B-exl3-4.5bpw", "t")
    assert calls and calls[0]["collection_slug"] == "owner/foo-abc"
    assert calls[0]["item_id"] == "owner/Foo-7B-exl3-4.5bpw"


def test_add_to_collection_noop_without_slug(monkeypatch):
    calls = []
    _fake_hf(monkeypatch, add_collection_item=lambda **kw: calls.append(kw))
    cards.add_to_collection("", "owner/whatever", "t")
    assert calls == []


def test_collection_url():
    assert cards.collection_url("owner/x").endswith("/collections/owner/x")
    assert cards.collection_url("") == ""


def test_pretty_title_override_wins():
    assert cards.pretty_title("foo-bar", "Custom · Title") == "Custom · Title"


def test_pretty_title_splits_on_separators():
    assert cards.pretty_title("Qwen3.6-35B-A3B") == "Qwen3.6 · 35B · A3B"


def test_derive_facts_moe():
    facts = cards.derive_model_facts(
        {"architectures": ["Qwen3MoeForCausalLM"], "num_hidden_layers": 40,
         "num_experts": 256},
        "Huihui-Qwen3.6-35B-A3B-abliterated",
    )
    assert facts["is_moe"] is True
    assert "40 layers" in facts["arch_line"]
    assert "256 experts" in facts["arch_line"]
    assert facts["arch_badge"] == "MoE_35B--A3B"  # shields.io doubles hyphens
    assert "mixture-of-experts" in facts["extra_tags"]
    assert "MoE expert batching" in facts["parallel_line"]


def test_derive_facts_dense():
    facts = cards.derive_model_facts(
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 32},
        "Llama-3.1-8B-Instruct",
    )
    assert facts["is_moe"] is False
    assert facts["arch_line"] == "Dense &nbsp;·&nbsp; 32 layers"
    assert facts["arch_badge"] == "Dense_8B"
    assert facts["extra_tags"] == ""


def test_quants_table_marks_current_and_queued():
    rows = [
        {"variant": "4.5", "head_bits": 8, "cal_rows": 250, "size_gb": 21.6,
         "url": "https://huggingface.co/x/y-exl3-4.5bpw"},
        {"variant": "4.0", "head_bits": 8, "cal_rows": 128, "size_gb": None,
         "url": None},
    ]
    table = cards.build_quants_table(rows, current_variant="4.5")
    # rows are bpw-sorted
    assert table.index("| 4.0 ") < table.index("**4.5**")
    assert "<kbd>this repo</kbd>" in table
    assert "<sub>queued</sub>" in table


def test_render_leaves_no_placeholders():
    card = cards.render_exl3_card(
        base_repo="huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated",
        repo_id="blockblockblock/Huihui-Qwen3.6-35B-A3B-abliterated-exl3-4.5bpw",
        variant="4.5",
        head_bits=8,
        cal_rows=250,
        size_gb=21.6,
        model_config={"architectures": ["Qwen3MoeForCausalLM"],
                      "num_hidden_layers": 40, "num_experts": 256},
        quant_rows=[{"variant": "4.5", "head_bits": 8, "cal_rows": 250,
                     "size_gb": 21.6, "url": None}],
        collection_url="https://huggingface.co/collections/blockblockblock/abc",
        license_id="apache-2.0",
        quantized_by="blockblockblock",
    )
    assert not re.search(r"\{\{[A-Z_]+\}\}", card), "unfilled placeholder left in card"
    assert "# Huihui · Qwen3.6 · 35B · A3B · abliterated" in card
    assert "license: apache-2.0" in card
    assert "bits_per_weight: 4.5" in card
    assert "blockblockblock/abc" in card
    assert "21.6&nbsp;GB" in card
