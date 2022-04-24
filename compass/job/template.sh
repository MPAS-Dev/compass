#!/bin/bash
#SBATCH  --job-name={{ job_name }}
{% if account != '' -%}
#SBATCH  --account={{ account}}
{%- endif %}
#SBATCH  --nodes={{ nodes }}
#SBATCH  --output={{ job_name }}.o%j
#SBATCH  --exclusive
#SBATCH  --time={{ wall_time }}
{% if qos != '' -%}
#SBATCH  --qos={{ qos }}
{%- endif %}
{% if partition != '' -%}
#SBATCH  --partition={{ partition }}
{%- endif %}
{% if constraint != '' -%}
#SBATCH  --constraint={{ constraint }}
{%- endif %}

source load_compass_env.sh
compass run

