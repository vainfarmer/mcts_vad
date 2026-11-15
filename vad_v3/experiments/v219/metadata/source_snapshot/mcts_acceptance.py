"""Experimental node-only acceptance, shared by live MCTS and frozen-tree replay."""
import math


NODE_ONLY_POLICIES = ('relaxed_node', 'force_best')


def select_node_proposal(nodes, baseline_quality, policy, max_quality_drop):
    """Pick one evaluated non-root node per candidate; never emit the whole tree.

    Nodes must already have passed the search geometry, evidence and response
    validation. Failed/unevaluated nodes cannot become a forced proposal.
    relaxed_node retains presence, event identity and the CLIP quality floor.
    force_best removes those acceptance gates as well as the final pair check.
    Both deliberately allow a reward below the root; ties keep search order.
    """
    if policy not in NODE_ONLY_POLICIES:
        raise ValueError('Not a node-only acceptance policy: ' + policy)
    eligible = []
    for node in nodes:
        ev = node.evaluation
        # force_best 也只能使用已成功评分的非根节点；失败响应不会被强制接收。
        if node.depth <= 0 or not ev or not math.isfinite(ev['reward']):
            continue
        if policy == 'relaxed_node':
            answer = ev['answer']
            quality = ev.get('parts', {}).get('C')
            if (not answer['target_present_inside'] or not answer['same_event_as_reference']
                    or quality is None or not math.isfinite(quality)
                    or quality < baseline_quality - max_quality_drop):
                continue
        eligible.append(node)
    # 每个候选只选一个最高奖励提案，不把整棵树中的区间都加入最终结果。
    return max(eligible, key=lambda node: node.evaluation['reward']) if eligible else None
