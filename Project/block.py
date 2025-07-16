from graphviz import Digraph

dot = Digraph(comment='PPO Anti-Jamming MIMO System', format='png')

# Nodes
dot.attr('node', shape='box', style='rounded,filled', color='lightgray')
dot.node('A', 'Channel Features\n+ Jammer Label')
dot.node('B', 'Random Forest\nJammer Classifier')
dot.node('C', 'PPO Agent\n(Modulation / Power / Nulling)')
dot.node('D', 'MATLAB MIMO+\nJamming Simulation')
dot.node('E', 'Reward:\nSINR - α·Pwr - β·BER')

# Arrows
dot.attr('edge', color='black')
dot.edge('A', 'B', label='Input Features')
dot.edge('A', 'C', label='Obs')
dot.edge('B', 'C', label='Predicted Jammer Type')
dot.edge('C', 'D', label='Action: [Mod, Pwr, Null]')
dot.edge('D', 'E', label='Output: SINR, BER')
dot.edge('E', 'C', label='Reward Feedback')

# Render and save
dot.render('ppo_anti_jamming_block_diagram', view=True)
