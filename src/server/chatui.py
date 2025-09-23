"""
chatui: chainlit based UI
"""
import json, functools, itertools
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable.utils import AddableDict
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage
from typing import cast
import chainlit as cl
from src.config import get_settings
from src.server.chain_cache import get_chat_chain
from src.chains.chains import chunk_dict_to_list
from src.utils.utils import omit
from langchain.globals import set_verbose, set_debug

# Get the environment
settings = get_settings()


@cl.on_chat_start
async def on_chat_start():
    # The priamry chain with tool call based routing
    runnable = get_chat_chain()
    cl.user_session.set("runnable", runnable)

    cl.user_session.set("messages", [])

    # Debug flags
    set_debug(True)
    set_verbose(False)


def make_markdown_table(data: list[list]|list[dict]):
    if len(data) == 0:
        return ""

    if isinstance(data[0], dict):
        # All rows should have same keys
        header = list(data[0].keys())
        data = [header] + [
            [r[c] for c in header]
            for r in data
        ]

    column_count = len(data[0])

    data = [
        [str("" if cell == None else cell) for cell in row]
        for row in data
    ]
    widths = [
        max(len(row[i]) for row in data)
        for i in range(column_count)
    ]

    data = [
        [f"{cell:<{width}}" for cell, width in zip(row, widths)]
        for row in data
    ]

    out = "|" + "|".join(data[0]) + "|\n"
    out += "|" + "|".join('-' * w for w in widths) + "|\n"
    for row in data[1:]:
        out += "|" + "|".join(row) + "|\n"

    return out


def bundled_custom_element(name, *, props, **kwargs):
    return cl.CustomElement(
        name = "CustomElement",
        props = { "element": name, "props": props },
        **kwargs,
    )


def get_tool_results(tool: str, messages: list[BaseMessage]) -> list:
    """ Gets ToolMessages for the given tool """
    return [
        msg
        for msg in messages
        if isinstance(msg, ToolMessage) and msg.name == tool
    ]


async def render_sql(response_msg: cl.Message, msgs: list[BaseMessage]):
    tool_msgs = get_tool_results("sql_qna_chain", msgs)

    # Display query results, if present
    if len(tool_msgs) > 0:
        # Assuming there's at most one query tool call
        tool_msg = tool_msgs[0]

        artifact = tool_msg.artifact or {}
        sql = None
        sql_result = None
        sql_error = None

        if tool_msg.status == "success":
            sql = artifact.get('query')
            sql_result = artifact.get('query_result')
        else:
            sql_error = artifact.get("error")

        if sql:
            response_msg.elements.append(cl.Text( # type: ignore
                name = "SQL Query",
                content = sql,
                display = "side",
            ))
        if sql_result:
            await response_msg.stream_token(
                "\n### SQL Query Results\n\n" +
                make_markdown_table(sql_result[:settings.query_output_limit_table]) + "\n"
            )
            # Need to add CustomElement so chainlit will add a link. Should make a PR or issue on
            # chainlit to separate the element name and the element component name so we can use a
            # more meaningful name.
            await response_msg.stream_token("Plot results: CustomElement\n")
            plot_element = bundled_custom_element("ConfigurablePlot",
                props = { "data": sql_result },
                display = "side",
            )
            response_msg.elements.append(plot_element) # type: ignore
        elif sql_error:
            await response_msg.stream_token(
                "\n### SQL Query Error\n\n" +
                "```\n" + 
                sql_error +
                "```\n"
            )


JOB_PRED_IN_COLUMNS = {
    "domain": "Domain",
    "node_count": "Node Count",
    "time_elapsed": "Time Elapsed",
    "utilization_type": "Utilization Type",
}

JOB_PRED_OUT_COLUMNS = {
    "stats_node_power_node_mean": "Node Power Mean",
    "stats_node_power_node_max": "Node Power Max",
    "stats_node_power_node_stddev": "Node Power Stddev",
    "stats_cpu_memory_power_node_max": "CPU Mem Power Max",
    "stats_gpu_power_node_max": "GPU Power Max",
    "stats_gpu_power_node_mean": "GPU Power Mean",
    "stats_gpu_power_node_stddev": "GPU Power Stddev",
    "stats_node_temp_node_max": "Node Temp Max",
    "stats_node_temp_node_stddev": "Node Temp Stddev",
    "stats_total_node_energy": "Total Energy",
    "stats_total_node_energy_node_max": "Node Energy Max",
    "stats_total_node_energy_node_mean": "Node Energy Mean",
    "stats_total_cpu_memory_energy": "CPU Mem Energy",
    "stats_total_gpu_energy": "GPU Energy",
    "stats_total_gpu_energy_node_max": "GPU Node Energy Max",
    "stats_total_gpu_energy_node_mean": "GPU Node Energy Mean",
}


async def render_job_pred_plot(response_msg: cl.Message, data: list[dict], out_cols: list[str]):
    # Just pick the first one, you can swap the out col in the widget
    out_col = out_cols[0]

    # find in_cols that have multiple values
    in_cols = [
        c for c in JOB_PRED_IN_COLUMNS.keys()
        if len(set(d[c] for d in data)) > 1
    ]
    main_group = in_cols[0] if len(in_cols) > 0 else "domain"
    secondary_groups = [c for c in in_cols if c != main_group]

    traces = []
    if len(secondary_groups) == 0:
        traces.append({
            "type": "bar",
            "mode": "markers",
            "orientation": "h",
            "y": [],
            "ysrc": main_group,
            "meta": {
                "columnNames": {"y": main_group, "x": out_col},
            },
            "x": [], # will be populated by the frontend component
            "xsrc": out_col,
        })
    else:
        data = [
            {**d, 'subgroup': '-'.join(d[c] for c in secondary_groups)}
            for d in data
        ]
        unique_subgroups = [*dict.fromkeys(d['subgroup'] for d in data)]

        for subgroup in unique_subgroups:
            traces.append({
                "type": "bar",
                "mode": "markers",
                "orientation": "h",
                "name": subgroup,
                "y": [],
                "ysrc": main_group,
                "meta": {
                    "columnNames": {"y": main_group, "x": out_col},
                },
                "x": [],
                "xsrc": out_col,
                "transforms": [
                    {
                        "type": "filter",
                        "target": [],
                        "targetsrc": "subgroup",
                        "meta": {
                            "columnNames": {
                                "target": "subgroup",
                            }
                        },
                        "value": subgroup,
                    }
                ]
            })

    fig = {
        "data": traces,
        "layout": {
            "xaxis": {
                "autorange": True,
                "type": "linear"
            },
            "yaxis": {
                "autorange": True,
                "type": "category"
            },
            "autosize": True,
        }
    }

    await response_msg.stream_token("Plot results: CustomElement\n")
    plot_element = bundled_custom_element("ConfigurablePlot",
        props = {
            "data": data,
            "initialFig": fig,
        },
        display = "side",
    )
    response_msg.elements.append(plot_element) # type: ignore
   

async def render_job_pred(response_msg: cl.Message, msgs: list[BaseMessage]):
    # Display job_pred results, if present
    tool_msgs = get_tool_results("job_pred", msgs)
    successful = [m.artifact['results'] for m in tool_msgs if m.status == 'success']
    failed = [m.content for m in tool_msgs if m.status != 'success']
    # Combine all job_pred results into single list
    result = list(itertools.chain(*successful))

    if len(tool_msgs) > 0:
        await response_msg.stream_token("\n### Prediction Results\n\n")

        if len(successful) > 0:

            data = result[:settings.query_output_limit_table]
            data = [{**omit(d, 'prediction'), **d['prediction']} for d in data]

            # Get all columns (dict.fromkeys to remove duplicates preserving order)
            used_out_columns = [*dict.fromkeys(itertools.chain(*[row.keys() for row in data]))]
            used_out_columns = [c for c in used_out_columns if c not in JOB_PRED_IN_COLUMNS]

            header = [
                ["Inputs"] + [""] * (len(JOB_PRED_IN_COLUMNS) - 1) + ["Predicted"] + [""] * (len(used_out_columns) - 1),
                [*JOB_PRED_IN_COLUMNS.values()] + [JOB_PRED_OUT_COLUMNS.get(c, c) for c in used_out_columns],
            ]

            table = header + [
                [d.get(c) for c in [*JOB_PRED_IN_COLUMNS.keys()] + used_out_columns]
                for d in data
            ]

            await response_msg.stream_token(make_markdown_table(table) + "\n")
            await render_job_pred_plot(response_msg, data, used_out_columns)

        if len(failed) > 0:
            await response_msg.stream_token("#### Errors\n")
            for error in failed:
                await response_msg.stream_token(error + "\n\n")


@cl.on_message
async def on_message(message: cl.Message):
    # Debug flags
    set_debug(True)
    set_verbose(False)

    # Get the runnable and message history
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    messages = cl.user_session.get("messages", [])[:] # copy
    # Add the current human message to the messsage history
    messages.append(HumanMessage(content = message.content))

    # Create the response message
    response_msg = cl.Message(content="")
    
    agent_stream = runnable.astream(
        messages,
        config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    )

    # the runnable streams AddableDict where each key is one of the message chunks returned
    # by the agent or the tool calls.
    agent_response_dict = AddableDict()
    async for chunk in agent_stream:
        agent_response_dict += chunk
        for msg_chunk in chunk_dict_to_list(chunk):
            if isinstance(msg_chunk, AIMessage):
                await response_msg.stream_token(msg_chunk.content)

    agent_response_messages = chunk_dict_to_list(agent_response_dict)
    messages = messages + agent_response_messages

    await render_sql(response_msg, agent_response_messages)
    await render_job_pred(response_msg, agent_response_messages)

    cl.user_session.set("messages", messages)
    await response_msg.send()
