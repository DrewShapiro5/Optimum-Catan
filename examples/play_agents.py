"""Run a game using the same agent registry as the web UI."""
from agent_apis import list_agents, register_agents
from catanatron import Game
from catanatron.registry import REGISTRY

def main():
    print(list_agents())
    register_agents()
    game=Game(REGISTRY.build_all(['JSETTLERS','JSETTLERS']),vps_to_win=15,discard_limit=9)
    print('Winner:',game.play())

if __name__=='__main__':main()
