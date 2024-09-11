from catanatron import Color
from catanatron.models.enums import (
    VICTORY_POINT,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    SETTLEMENT,
    CITY,
    ROAD
)
from catanatron.models.map import get_nodes_and_edges
from catanatron.models.coordinate_system import add

def step_reward(game,p0_color):# nump turns in games are in hundreds. So num_turns/100 gives decimal reward
    step_reward = float(game.state.num_turns/1000)
    return float(-1.0*step_reward)

def city_reward(game,p0_color):
    p0_vp = game.state.player_state["P0_VICTORY_POINTS"]
    cities_left= game.state.player_state["P0_CITIES_AVAILABLE"]
    cities_built = 4 - cities_left
    reward = (p0_vp*(1/3)) + (cities_built*(2/3))
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is None:
        return reward *0.0001
    else:
        return -1
    
def settlement_reward(game, p0_color):
    p0_vp = game.state.player_state["P0_VICTORY_POINTS"]
    settlements_left= game.state.player_state["P0_SETTLEMENTS_AVAILABLE"]
    settlements_built = 5 - settlements_left
    reward = (p0_vp*(1/3)) + (settlements_built*(2/3))
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is None:
        return reward *0.0001
    else:
        return -1

def longest_road_reward(game, p0_color):
    p0_vp = game.state.player_state["P0_VICTORY_POINTS"]
    roads_left= game.state.player_state["P0_ROADS_AVAILABLE"]
    roads_built = 15 - roads_left
    has_road = 0
    #Has Longest road 
    if game.state.player_state["P0_HAS_ROAD"]:
        has_road = 1
    reward = (p0_vp*(1/3)) + (roads_built*(1/3)) +(has_road*(1/3))
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is None:
        return reward *0.0001
    else:
        return 1
    
def dev_card_rewards(game, p0_color):
    index_self = str(game.state.current_player_index)
    index_enemy =  "1" if index_self == "0" else "0"
    color_self =  list(game.state.color_to_index.keys())[list(game.state.color_to_index.values()).index(int(index_self))]
    color_enemy =  list(game.state.color_to_index.keys())[list(game.state.color_to_index.values()).index(int(index_enemy))]
	
    p_state = game.state.player_state
    board = game.state.board
    # is winner
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is not None:
        return -1
    reward = 0
    # + for dev cards and more for played dev cards (to disencourage hoarding)
    reward += (2*(p_state["P"+index_self+"_KNIGHT_IN_HAND"] + p_state["P"+index_self+"_YEAR_OF_PLENTY_IN_HAND"] + p_state["P"+index_self+"_MONOPOLY_IN_HAND"] + p_state["P"+index_self+"_ROAD_BUILDING_IN_HAND"])
				+ 4*p_state["P"+index_self+"_PLAYED_KNIGHT"] + 3*(p_state["P"+index_self+"_PLAYED_YEAR_OF_PLENTY"] + p_state["P"+index_self+"_PLAYED_MONOPOLY"] + p_state["P"+index_self+"_PLAYED_ROAD_BUILDING"])
				+ 2*p_state["P"+index_self+"_VICTORY_POINT_IN_HAND"] + 8*p_state["P"+index_self+"_PLAYED_VICTORY_POINT"])
    reward -= (2*(p_state["P"+index_enemy+"_KNIGHT_IN_HAND"] + p_state["P"+index_enemy+"_YEAR_OF_PLENTY_IN_HAND"] + p_state["P"+index_enemy+"_MONOPOLY_IN_HAND"] + p_state["P"+index_enemy+"_ROAD_BUILDING_IN_HAND"])
				+ 4*p_state["P"+index_enemy+"_PLAYED_KNIGHT"] + 3*(p_state["P"+index_enemy+"_PLAYED_YEAR_OF_PLENTY"] + p_state["P"+index_enemy+"_PLAYED_MONOPOLY"] + p_state["P"+index_enemy+"_PLAYED_ROAD_BUILDING"])
				+ 2*p_state["P"+index_enemy+"_VICTORY_POINT_IN_HAND"] + 8*p_state["P"+index_enemy+"_PLAYED_VICTORY_POINT"]) # bot doesn't know whether enemy's unplayed card is a victory point
    return reward / 100000

def complex_reward(game, p0_color):
    p0_vp = game.state.player_state["P0_VICTORY_POINTS"]
    has_road = 0
    if game.state.player_state["P0_HAS_ROAD"]:
        has_road = 1
    has_army = 0
    if game.state.player_state["P0_HAS_ARMY"]:
        has_army = 1
    winning_color = game.winning_color()
    # step_rewards = step_reward(game,p0_color)
    reward =0
    reward = p0_vp #+ has_road + has_army #+ step_rewards
    # print(p0_vp , has_road , has_army)
    if p0_color == winning_color:
        return 1
    elif winning_color is None:
        return reward*0.0001
    else:
        return -1
    
def reward_function(game, p0_color):
	index_self = str(game.state.current_player_index)
	index_enemy =  "1" if index_self == "0" else "0"
	color_self =  list(game.state.color_to_index.keys())[list(game.state.color_to_index.values()).index(int(index_self))]
	color_enemy =  list(game.state.color_to_index.keys())[list(game.state.color_to_index.values()).index(int(index_enemy))]
	
	p_state = game.state.player_state
	board = game.state.board

	# is winner
	winning_color = game.winning_color()
	if p0_color == winning_color:
		return 1
	elif winning_color is not None:
		return -1
	
	# will be divided by 10000
	reward = 0

	# + for each possessed resource, but - if too many; - for enemy cards ;; removed because of hoarding
	resource_amount = p_state["P"+index_self+"_WOOD_IN_HAND"] + p_state["P"+index_self+"_BRICK_IN_HAND"] + p_state["P"+index_self+"_SHEEP_IN_HAND"] + p_state["P"+index_self+"_WHEAT_IN_HAND"] + p_state["P"+index_self+"_ORE_IN_HAND"]
	# reward += 0.5*resource_amount
	if (resource_amount > 7):
		reward -= 2*(resource_amount - 7)

	# enemy_resource_amount = p_state["P"+index_enemy+"_WOOD_IN_HAND"] + p_state["P"+index_enemy+"_BRICK_IN_HAND"] + p_state["P"+index_enemy+"_SHEEP_IN_HAND"] + p_state["P"+index_enemy+"_WHEAT_IN_HAND"] + p_state["P"+index_enemy+"_ORE_IN_HAND"]
	# reward -= 0.5*enemy_resource_amount

	# + for dev cards and more for played dev cards (to disencourage hoarding)
	reward += (2*(p_state["P"+index_self+"_KNIGHT_IN_HAND"] + p_state["P"+index_self+"_YEAR_OF_PLENTY_IN_HAND"] + p_state["P"+index_self+"_MONOPOLY_IN_HAND"] + p_state["P"+index_self+"_ROAD_BUILDING_IN_HAND"])
				+ 4*p_state["P"+index_self+"_PLAYED_KNIGHT"] + 3*(p_state["P"+index_self+"_PLAYED_YEAR_OF_PLENTY"] + p_state["P"+index_self+"_PLAYED_MONOPOLY"] + p_state["P"+index_self+"_PLAYED_ROAD_BUILDING"])
				+ 2*p_state["P"+index_self+"_VICTORY_POINT_IN_HAND"] + 8*p_state["P"+index_self+"_PLAYED_VICTORY_POINT"])
	reward -= (2*(p_state["P"+index_enemy+"_KNIGHT_IN_HAND"] + p_state["P"+index_enemy+"_YEAR_OF_PLENTY_IN_HAND"] + p_state["P"+index_enemy+"_MONOPOLY_IN_HAND"] + p_state["P"+index_enemy+"_ROAD_BUILDING_IN_HAND"])
				+ 4*p_state["P"+index_enemy+"_PLAYED_KNIGHT"] + 3*(p_state["P"+index_enemy+"_PLAYED_YEAR_OF_PLENTY"] + p_state["P"+index_enemy+"_PLAYED_MONOPOLY"] + p_state["P"+index_enemy+"_PLAYED_ROAD_BUILDING"])
				+ 2*p_state["P"+index_enemy+"_VICTORY_POINT_IN_HAND"] + 8*p_state["P"+index_enemy+"_PLAYED_VICTORY_POINT"]) # bot doesn't know whether enemy's unplayed card is a victory point

	# + for robber on enemy tile, - for robber on owned tile
	robbed_nodes = board.map.tiles[board.robber_coordinate].nodes
	for node_id in robbed_nodes.values():
		if not node_id in list(board.buildings.keys()):
			continue
		robbed_building = board.buildings[node_id]
		reward += -10 if robbed_building[0] == color_self else 10
		
	# + per vp difference
	reward += 50 * (p_state["P"+index_self+"_VICTORY_POINTS"] - p_state["P"+index_enemy+"_VICTORY_POINTS"])

	# +++ per city & ++ per village & + per road; city and village already in vp?
	reward += 10*len(game.state.buildings_by_color[color_self][ROAD])
	reward += 50*len(game.state.buildings_by_color[color_self][SETTLEMENT])
	reward += 150*len(game.state.buildings_by_color[color_self][CITY])

	reward -= 10*len(game.state.buildings_by_color[color_enemy][ROAD])
	reward -= 50*len(game.state.buildings_by_color[color_enemy][SETTLEMENT])
	reward -= 150*len(game.state.buildings_by_color[color_enemy][CITY])

	return reward / 10000

def VP_only_reward_function(game, p0_color):
	index_self = str(game.state.current_player_index)
	index_enemy =  "1" if index_self == "0" else "0"
	p_state = game.state.player_state

	# is winner
	winning_color = game.winning_color()
	if p0_color == winning_color:
		return 1
	elif winning_color is not None:
		return -1
	# victory point difference
	return 0.01 * (p_state["P"+index_self+"_VICTORY_POINTS"] - p_state["P"+index_enemy+"_VICTORY_POINTS"])