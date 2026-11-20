import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Scikit-Learn & NumPy Compatibility Layer
# -----------------------------------------------------------------------------
# 1. Patch NumPy BitGenerator constructor for cross-version (NumPy 1.x vs 2.x) pickle compatibility
try:
    import numpy.random._pickle as npr_pickle
    orig_ctor = npr_pickle.__bit_generator_ctor
    def safe_bit_generator_ctor(bit_generator="MT19937"):
        if not isinstance(bit_generator, str):
            if hasattr(bit_generator, "__name__"):
                bit_generator = bit_generator.__name__
            else:
                bit_generator = str(bit_generator).split(".")[-1].replace("'>", "")
        return orig_ctor(bit_generator)
    npr_pickle.__bit_generator_ctor = safe_bit_generator_ctor
except Exception:
    pass

# 2. Ensure Cython _loss references from scikit-learn unpickle correctly across versions
try:
    import sklearn._loss._loss
    sys.modules["_loss"] = sklearn._loss._loss
except Exception:
    pass

class RobustModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "_loss":
            try:
                import sklearn._loss._loss as sk_loss
                return getattr(sk_loss, name)
            except Exception:
                pass
        if module == "sklearn._loss.loss":
            try:
                import sklearn._loss as sk_l
                if hasattr(sk_l, name):
                    return getattr(sk_l, name)
            except Exception:
                pass
        return super().find_class(module, name)

# Model Initialization
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Gradient_model.pkl")
model = None
load_debug = {
    "base_dir": BASE_DIR,
    "model_path": MODEL_PATH,
    "model_exists": os.path.exists(MODEL_PATH),
    "files_in_dir": os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else [],
}

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = RobustModelUnpickler(f).load()
        load_debug["status"] = "loaded_via_robust"
    except Exception as e:
        load_debug["robust_error"] = str(e)
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            load_debug["status"] = "loaded_via_standard"
        except Exception as e2:
            load_debug["standard_error"] = str(e2)
else:
    load_debug["status"] = "model_file_not_found"

# -----------------------------------------------------------------------------
# Feature Schema and Metadata
# -----------------------------------------------------------------------------
FEATURE_METADATA = [
    {
        "key": "Sales",
        "label": "Gross Sales ($)",
        "group": "financial",
        "min": 1,
        "max": 10000,
        "step": 1,
        "default": 250.00,
        "type": "number",
    },
    {
        "key": "Discount",
        "label": "Discount Applied",
        "group": "financial",
        "min": 0,
        "max": 0.8,
        "step": 0.01,
        "default": 0.10,
        "type": "number",
    },
    {
        "key": "Quantity",
        "label": "Units Ordered",
        "group": "financial",
        "min": 1,
        "max": 50,
        "step": 1,
        "default": 3,
        "type": "number",
    },
    {
        "key": "Ship Mode",
        "label": "Shipping Speed",
        "group": "logistics",
        "default": 0,
        "type": "select",
        "options": [
            {"value": 0, "label": "First Class (Express 1-2 Days)"},
            {"value": 1, "label": "Same Day (Priority Expedited)"},
            {"value": 2, "label": "Second Class (Standard Expedited)"},
            {"value": 3, "label": "Standard Class (Economy Ground)"},
        ],
    },
    {
        "key": "Category",
        "label": "Product Category",
        "group": "catalog",
        "default": 0,
        "type": "select",
        "options": [
            {"value": 0, "label": "Furniture (Chairs, Tables, Bookcases)"},
            {"value": 1, "label": "Office Supplies (Paper, Binders, Art)"},
            {"value": 2, "label": "Technology (Phones, Machines, Copiers)"},
        ],
    },
    {
        "key": "Sub-Category",
        "label": "Product Sub-Category",
        "group": "catalog",
        "default": 3,
        "type": "select",
        "options": [
            {"value": 0, "label": "Accessories"},
            {"value": 1, "label": "Appliances"},
            {"value": 2, "label": "Art"},
            {"value": 3, "label": "Binders"},
            {"value": 4, "label": "Bookcases"},
            {"value": 5, "label": "Chairs"},
            {"value": 6, "label": "Copiers"},
            {"value": 7, "label": "Envelopes"},
            {"value": 8, "label": "Fasteners"},
            {"value": 9, "label": "Furnishings"},
            {"value": 10, "label": "Labels"},
            {"value": 11, "label": "Machines"},
            {"value": 12, "label": "Paper"},
            {"value": 13, "label": "Phones"},
            {"value": 14, "label": "Storage"},
            {"value": 15, "label": "Supplies"},
            {"value": 16, "label": "Tables"},
        ],
    },
    {
        "key": "Segment",
        "label": "Market Segment",
        "group": "customer",
        "default": 0,
        "type": "select",
        "options": [
            {"value": 0, "label": "Consumer (Individual Retail)"},
            {"value": 1, "label": "Corporate (B2B Commercial)"},
            {"value": 2, "label": "Home Office (Small Business)"},
        ],
    },
    {
        "key": "Region",
        "label": "Sales Territory",
        "group": "logistics",
        "default": 1,
        "type": "select",
        "options": [
            {"value": 0, "label": "Central Region"},
            {"value": 1, "label": "East Region"},
            {"value": 2, "label": "South Region"},
            {"value": 3, "label": "West Region"},
        ],
    },
    {
        "key": "State",
        "label": "State Code Index",
        "group": "logistics",
        "min": 0,
        "max": 48,
        "step": 1,
        "default": 10,
        "type": "range",
    },
    {
        "key": "City",
        "label": "Metro Area Index",
        "group": "logistics",
        "min": 0,
        "max": 500,
        "step": 1,
        "default": 20,
        "type": "range",
    },
    {
        "key": "Customer Name",
        "label": "Account ID Index",
        "group": "customer",
        "min": 0,
        "max": 792,
        "step": 1,
        "default": 10,
        "type": "range",
    },
    {
        "key": "Product Name",
        "label": "SKU Identifier",
        "group": "catalog",
        "min": 0,
        "max": 1849,
        "step": 1,
        "default": 100,
        "type": "range",
    },
    {
        "key": "Country",
        "label": "Country Origin",
        "group": "logistics",
        "min": 0,
        "max": 0,
        "step": 1,
        "default": 0,
        "type": "hidden",
    },
]

DEFAULT_VALUES = {item["key"]: item["default"] for item in FEATURE_METADATA}
FEATURE_ORDER = [
    "Ship Mode",
    "Customer Name",
    "Segment",
    "Country",
    "City",
    "State",
    "Region",
    "Category",
    "Sub-Category",
    "Product Name",
    "Sales",
    "Quantity",
    "Discount",
]

PRESET_SCENARIOS = {
    "standard": {
        "name": "Standard Office Bulk",
        "desc": "Typical B2B order of binders and desk accessories with standard 10% discount.",
        "values": {
            "Sales": 250.0,
            "Discount": 0.10,
            "Quantity": 3,
            "Ship Mode": 0,
            "Category": 1,
            "Sub-Category": 3,
            "Segment": 1,
            "Region": 1,
            "State": 10,
            "City": 20,
            "Customer Name": 10,
            "Product Name": 100,
            "Country": 0,
        },
    },
    "tech_high_margin": {
        "name": "High-Value Technology Order",
        "desc": "Premium Phones / Copiers sale with 0% discount resulting in strong profitability.",
        "values": {
            "Sales": 1500.0,
            "Discount": 0.00,
            "Quantity": 2,
            "Ship Mode": 1,
            "Category": 2,
            "Sub-Category": 13,
            "Segment": 0,
            "Region": 3,
            "State": 3,
            "City": 50,
            "Customer Name": 25,
            "Product Name": 250,
            "Country": 0,
        },
    },
    "deep_discount_loss": {
        "name": "High-Discount Margin Dilutive",
        "desc": "Heavy clearance sale with 70% discount causing negative operational margins.",
        "values": {
            "Sales": 380.0,
            "Discount": 0.70,
            "Quantity": 4,
            "Ship Mode": 3,
            "Category": 0,
            "Sub-Category": 16,
            "Segment": 0,
            "Region": 0,
            "State": 15,
            "City": 80,
            "Customer Name": 45,
            "Product Name": 120,
            "Country": 0,
        },
    },
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NexusProfit Enterprise — Margin Forecast & Decision Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Plus Jakarta Sans', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        slate: { 950: '#020617' }
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-[#0b0f17] text-slate-100 min-h-screen antialiased font-sans selection:bg-cyan-500 selection:text-black">
    <div x-data="marginWorkspace()" x-init="init()" class="flex flex-col min-h-screen">
        <!-- Enterprise App Bar -->
        <header class="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
            <div class="max-w-[1600px] mx-auto px-6 h-16 flex items-center justify-between">
                <div class="flex items-center space-x-4">
                    <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-black text-lg shadow-lg shadow-cyan-500/20">
                        N
                    </div>
                    <div>
                        <div class="flex items-center space-x-2">
                            <span class="font-bold tracking-tight text-white text-base">NexusProfit</span>
                            <span class="bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-[10px] px-1.5 py-0.5 rounded uppercase font-mono tracking-wider font-semibold">Enterprise</span>
                        </div>
                        <p class="text-[11px] text-slate-400">Gradient-Boosted Retail Profit Engine • Superstore Analytics</p>
                    </div>
                </div>

                <div class="flex items-center space-x-4 text-xs">
                    <div class="hidden sm:flex items-center space-x-2 bg-slate-950/70 border border-slate-800 rounded-full px-3 py-1 font-mono text-slate-400">
                        <span class="w-2 h-2 rounded-full" :class="modelOnline ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'"></span>
                        <span x-text="modelOnline ? 'GBR Model (Active ML)' : 'Simulation Mode'"></span>
                    </div>
                    <button @click="resetDefaults()" class="text-slate-400 hover:text-white transition py-1.5 px-3 rounded-md hover:bg-slate-800 border border-transparent hover:border-slate-700">
                        Reset Defaults
                    </button>
                    <button @click="runSimulation()" class="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold px-4 py-1.5 rounded-lg transition text-xs shadow-lg shadow-cyan-500/20">
                        Recalculate
                    </button>
                </div>
            </div>
        </header>

        <!-- Main Studio Body -->
        <main class="max-w-[1600px] mx-auto p-6 flex-1 w-full grid grid-cols-1 xl:grid-cols-12 gap-6">
            <!-- Left Config Deck -->
            <div class="xl:col-span-5 flex flex-col space-y-4">
                <!-- Preset Scenarios -->
                <div class="bg-slate-900/80 border border-slate-800/90 rounded-xl p-4">
                    <div class="text-xs font-mono uppercase tracking-wider text-slate-400 mb-2">Preset Retail Scenarios</div>
                    <div class="grid grid-cols-3 gap-2">
                        <button @click="loadPreset('standard')" class="p-2 text-left rounded-lg border border-slate-800 bg-slate-950/70 hover:border-cyan-500/60 transition text-xs">
                            <span class="font-bold text-white block truncate">Office Bulk</span>
                            <span class="text-[10px] text-slate-400">$250 • 10% Disc</span>
                        </button>
                        <button @click="loadPreset('tech_high_margin')" class="p-2 text-left rounded-lg border border-slate-800 bg-slate-950/70 hover:border-cyan-500/60 transition text-xs">
                            <span class="font-bold text-emerald-400 block truncate">Tech Premium</span>
                            <span class="text-[10px] text-slate-400">$1,500 • 0% Disc</span>
                        </button>
                        <button @click="loadPreset('deep_discount_loss')" class="p-2 text-left rounded-lg border border-slate-800 bg-slate-950/70 hover:border-rose-500/60 transition text-xs">
                            <span class="font-bold text-rose-400 block truncate">Loss-Leader</span>
                            <span class="text-[10px] text-slate-400">$380 • 70% Disc</span>
                        </button>
                    </div>
                </div>

                <div class="bg-slate-900/80 border border-slate-800/90 rounded-xl p-5 shadow-sm">
                    <div class="flex items-center justify-between pb-4 border-b border-slate-800">
                        <h2 class="text-sm font-semibold tracking-wide text-white uppercase font-mono">Store Parameters</h2>
                        <div class="flex space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                            <button @click="activeTab = 'financial'" :class="activeTab === 'financial' ? 'bg-slate-800 text-cyan-400 font-medium' : 'text-slate-400'" class="px-2.5 py-1 rounded transition">Financial</button>
                            <button @click="activeTab = 'catalog'" :class="activeTab === 'catalog' ? 'bg-slate-800 text-cyan-400 font-medium' : 'text-slate-400'" class="px-2.5 py-1 rounded transition">Catalog</button>
                            <button @click="activeTab = 'customer'" :class="activeTab === 'customer' ? 'bg-slate-800 text-cyan-400 font-medium' : 'text-slate-400'" class="px-2.5 py-1 rounded transition">Customer</button>
                            <button @click="activeTab = 'logistics'" :class="activeTab === 'logistics' ? 'bg-slate-800 text-cyan-400 font-medium' : 'text-slate-400'" class="px-2.5 py-1 rounded transition">Logistics</button>
                        </div>
                    </div>

                    <form @submit.prevent="runSimulation()" class="mt-4 space-y-4">
                        <template x-for="item in activeFields" :key="item.key">
                            <div x-show="item.type !== 'hidden'" class="bg-slate-950/60 p-3.5 rounded-lg border border-slate-800/70 hover:border-slate-700/80 transition">
                                <div class="flex justify-between items-center mb-2">
                                    <label :for="item.key" class="text-xs font-medium text-slate-300" x-text="item.label"></label>
                                    <span class="font-mono text-xs text-cyan-400 font-semibold" x-text="formatDisplay(item.key, formData[item.key])"></span>
                                </div>

                                <template x-if="item.type === 'select'">
                                    <select 
                                        :id="item.key"
                                        x-model.number="formData[item.key]"
                                        @change="runSimulation()"
                                        class="w-full bg-slate-900 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-100 font-mono focus:border-cyan-500 focus:outline-none"
                                    >
                                        <template x-for="opt in item.options" :key="opt.value">
                                            <option :value="opt.value" x-text="opt.label" :selected="opt.value === formData[item.key]"></option>
                                        </template>
                                    </select>
                                </template>

                                <template x-if="item.type === 'number'">
                                    <input 
                                        type="number"
                                        :min="item.min"
                                        :max="item.max"
                                        :step="item.step"
                                        :id="item.key"
                                        x-model.number="formData[item.key]"
                                        @input.debounce.250ms="runSimulation()"
                                        class="w-full accent-cyan-500 bg-slate-900 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-100 font-mono focus:border-cyan-500 focus:outline-none"
                                    >
                                </template>

                                <template x-if="item.type === 'range'">
                                    <input 
                                        type="range"
                                        :min="item.min"
                                        :max="item.max"
                                        :step="item.step"
                                        :id="item.key"
                                        x-model.number="formData[item.key]"
                                        @input.debounce.250ms="runSimulation()"
                                        class="w-full accent-cyan-500 bg-slate-900 border border-slate-800 rounded px-2 py-1 text-sm text-slate-100 font-mono focus:border-cyan-500 focus:outline-none"
                                    >
                                </template>
                            </div>
                        </template>

                        <button 
                            type="submit" 
                            :disabled="loading"
                            class="w-full mt-2 py-2.5 px-4 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-bold text-xs uppercase tracking-wider hover:opacity-95 transition flex items-center justify-center space-x-2 shadow-lg shadow-cyan-500/20"
                        >
                            <span x-show="loading" class="animate-spin h-3.5 w-3.5 border-2 border-slate-950 border-t-transparent rounded-full"></span>
                            <span x-text="loading ? 'Recalculating Margin Matrix...' : 'Run Scenario Estimation'"></span>
                        </button>
                    </form>
                </div>
            </div>

            <!-- Right Analytics Matrix -->
            <div class="xl:col-span-7 flex flex-col space-y-6">
                <!-- KPI Executive Row -->
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div class="bg-slate-900/80 border border-slate-800/90 rounded-xl p-5 relative overflow-hidden">
                        <div class="text-[11px] font-mono uppercase text-slate-400 tracking-wider">Projected Net Profit</div>
                        <div class="text-3xl font-extrabold tracking-tight mt-1 font-mono" :class="prediction >= 0 ? 'text-emerald-400' : 'text-rose-400'">
                            <span x-text="prediction !== null ? (prediction >= 0 ? `$${prediction.toFixed(2)}` : `-$${Math.abs(prediction).toFixed(2)}`) : '---'"></span>
                        </div>
                        <div class="mt-2 text-[11px] text-slate-500 flex items-center space-x-1.5">
                            <span class="inline-block w-1.5 h-1.5 rounded-full" :class="prediction >= 0 ? 'bg-emerald-400' : 'bg-rose-400'"></span>
                            <span x-text="prediction >= 0 ? 'Accretive transaction' : 'Margin dilutive transaction'"></span>
                        </div>
                    </div>

                    <div class="bg-slate-900/80 border border-slate-800/90 rounded-xl p-5">
                        <div class="text-[11px] font-mono uppercase text-slate-400 tracking-wider">Profit Margin Ratio</div>
                        <div class="text-3xl font-extrabold tracking-tight mt-1 text-white font-mono" :class="marginRatio >= 0 ? 'text-white' : 'text-rose-400'">
                            <span x-text="marginRatio !== null ? `${marginRatio.toFixed(1)}%` : '---'"></span>
                        </div>
                        <div class="mt-2 text-[11px] text-slate-500">Based on gross realized sales</div>
                    </div>

                    <div class="bg-slate-900/80 border border-slate-800/90 rounded-xl p-5">
                        <div class="text-[11px] font-mono uppercase text-slate-400 tracking-wider">Breakeven Factor</div>
                        <div class="text-3xl font-extrabold tracking-tight mt-1 text-cyan-400 font-mono">
                            <span x-text="breakevenThreshold !== null ? `$${breakevenThreshold.toFixed(2)}` : '---'"></span>
                        </div>
                        <div class="mt-2 text-[11px] text-slate-500">Cost/Absorption floor estimate</div>
                    </div>
                </div>

                <!-- Waterfall Chart and Analytics -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1">
                    <div class="bg-slate-900/80 border border-slate-800/90 rounded-xl p-5 flex flex-col">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-xs font-mono uppercase tracking-wider text-slate-300">Sensitivity Analysis (Discount Curve)</h3>
                        </div>
                        <div class="relative flex-1 min-h-[240px]">
                            <canvas id="sensitivityChart"></canvas>
                        </div>
                    </div>

                    <div class="bg-slate-900/80 border border-slate-800/90 rounded-xl p-5 flex flex-col">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-xs font-mono uppercase tracking-wider text-slate-300">Revenue & Yield Distribution</h3>
                        </div>
                        <div class="relative flex-1 min-h-[240px]">
                            <canvas id="distributionChart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Structured Metadata Audit Output -->
                <div class="bg-slate-900/80 border border-slate-800/90 rounded-xl p-5">
                    <h3 class="text-xs font-mono uppercase tracking-wider text-slate-300 mb-3">Model Vector Telemetry</h3>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
                        <div class="bg-slate-950 p-2.5 rounded border border-slate-800/70">
                            <span class="text-slate-500 block text-[10px]">TOTAL FEATURES</span>
                            <span class="text-slate-200 font-bold">13 Variables</span>
                        </div>
                        <div class="bg-slate-950 p-2.5 rounded border border-slate-800/70">
                            <span class="text-slate-500 block text-[10px]">ALGORITHM</span>
                            <span class="text-cyan-400 font-bold">GBR-100 Trees</span>
                        </div>
                        <div class="bg-slate-950 p-2.5 rounded border border-slate-800/70">
                            <span class="text-slate-500 block text-[10px]">EFFECTIVE DISCOUNT</span>
                            <span class="text-slate-200 font-bold" x-text="`${(formData.Discount * 100).toFixed(0)}%`"></span>
                        </div>
                        <div class="bg-slate-950 p-2.5 rounded border border-slate-800/70">
                            <span class="text-slate-500 block text-[10px]">AVG BASKET UNIT PRICE</span>
                            <span class="text-slate-200 font-bold" x-text="`$${(formData.Sales / Math.max(1, formData.Quantity)).toFixed(2)}`"></span>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('marginWorkspace', () => ({
                activeTab: 'financial',
                loading: false,
                modelOnline: {{ 'true' if model_loaded else 'false' }},
                prediction: null,
                marginRatio: null,
                breakevenThreshold: null,
                fields: JSON.parse('{{ metadata | tojson | safe }}'),
                formData: JSON.parse('{{ default_values | tojson | safe }}'),
                presets: JSON.parse('{{ presets | tojson | safe }}'),
                sensitivityChart: null,
                distributionChart: null,

                get activeFields() {
                    return this.fields.filter(f => f.group === this.activeTab);
                },

                init() {
                    this.$nextTick(() => {
                        this.initCharts();
                        this.runSimulation();
                    });
                },

                formatDisplay(key, val) {
                    if (key === 'Sales') return `$${Number(val).toFixed(2)}`;
                    if (key === 'Discount') return `${(Number(val) * 100).toFixed(0)}%`;
                    if (key === 'Quantity') return `${val} units`;
                    const f = this.fields.find(item => item.key === key);
                    if (f && f.options) {
                        const match = f.options.find(o => o.value == val);
                        if (match) return match.label.split(' (')[0];
                    }
                    return `${val}`;
                },

                loadPreset(name) {
                    if (this.presets[name]) {
                        this.formData = Object.assign({}, this.presets[name].values);
                        this.runSimulation();
                    }
                },

                resetDefaults() {
                    this.formData = JSON.parse('{{ default_values | tojson | safe }}');
                    this.runSimulation();
                },

                async runSimulation() {
                    this.loading = true;
                    try {
                        const res = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ features: this.formData })
                        });
                        const data = await res.json();
                        if (data.status === 'success') {
                            this.prediction = data.prediction;
                            this.marginRatio = (this.prediction / Math.max(1, this.formData.Sales)) * 100;
                            this.breakevenThreshold = Math.max(0, this.formData.Sales - this.prediction);
                            this.updateCharts();
                        }
                    } catch (err) {
                        console.error('Inference pipeline error:', err);
                    } finally {
                        this.loading = false;
                    }
                },

                initCharts() {
                    const ctxSens = document.getElementById('sensitivityChart').getContext('2d');
                    this.sensitivityChart = new Chart(ctxSens, {
                        type: 'line',
                        data: {
                            labels: ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%'],
                            datasets: [{
                                label: 'Projected Profit vs Discount',
                                data: [0, 0, 0, 0, 0, 0, 0, 0],
                                borderColor: '#06b6d4',
                                backgroundColor: 'rgba(6, 182, 212, 0.08)',
                                fill: true,
                                tension: 0.25,
                                borderWidth: 2,
                                pointRadius: 3
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } } },
                                x: { grid: { display: false }, ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } } }
                            }
                        }
                    });

                    const ctxDist = document.getElementById('distributionChart').getContext('2d');
                    this.distributionChart = new Chart(ctxDist, {
                        type: 'doughnut',
                        data: {
                            labels: ['Estimated Cost Base', 'Retained Net Margin'],
                            datasets: [{
                                data: [80, 20],
                                backgroundColor: ['#334155', '#06b6d4'],
                                borderWidth: 0
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'bottom',
                                    labels: { color: '#94a3b8', boxWidth: 10, font: { family: 'JetBrains Mono', size: 10 } }
                                }
                            },
                            cutout: '72%'
                        }
                    });
                },

                updateCharts() {
                    if (!this.sensitivityChart || !this.distributionChart) return;

                    const baseYield = this.prediction !== null ? this.prediction : 42.5;
                    const steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
                    const curve = steps.map(d => {
                        const delta = (this.formData.Discount - d) * this.formData.Sales * 0.85;
                        return Math.round((baseYield + delta) * 100) / 100;
                    });

                    this.sensitivityChart.data.datasets[0].data = curve;
                    this.sensitivityChart.update();

                    const cost = Math.max(0, this.formData.Sales - (this.prediction || 0));
                    const profit = Math.max(0, this.prediction || 0);

                    this.distributionChart.data.datasets[0].data = [cost, profit];
                    this.distributionChart.update();
                }
            }));
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        default_values=DEFAULT_VALUES,
        metadata=FEATURE_METADATA,
        presets=PRESET_SCENARIOS,
        model_loaded=(model is not None),
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            data = {}
        input_dict = data.get("features", {})
        input_vector = [float(input_dict.get(feat, DEFAULT_VALUES.get(feat, 0))) for feat in FEATURE_ORDER]

        if model is not None:
            # Use DataFrame with feature names to suppress sklearn UserWarning
            df_input = pd.DataFrame([input_vector], columns=FEATURE_ORDER)
            pred = float(model.predict(df_input)[0])
        else:
            # Deterministic linear fallback preserving retail margin dynamics
            sales = float(input_dict.get("Sales", 250))
            discount = float(input_dict.get("Discount", 0.1))
            pred = float(sales * (0.28 - (discount * 0.5)))

        return jsonify({
            "status": "success",
            "prediction": round(pred, 2),
            "model_type": "GradientBoostingRegressor" if model is not None else "Simulation",
            "debug": load_debug
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
