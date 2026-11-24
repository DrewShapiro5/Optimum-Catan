import com.google.gson.*;
import java.io.*;
import java.util.*;
import soc.baseclient.ServerConnectInfo;
import soc.game.*;
import soc.message.SOCMessage;
import soc.message.SOCRejectOffer;
import soc.robot.*;
import soc.util.*;

/**
 * Runs upstream SOCRobotBrain against a persistent, client-side SOCGame. Transport and
 * rule/coordinate translation belong to the adapter, not the brain.
 */
public final class Bridge {
  static final Gson JSON = new Gson();
  static final Map<String, Session> sessions = new LinkedHashMap<String, Session>(16, .75f, true);

  static int integer(JsonObject o, String k) {
    return o.get(k).getAsInt();
  }

  static int[] ints(JsonArray a) {
    int[] v = new int[a.size()];
    for (int i = 0; i < v.length; i++) v[i] = a.get(i).getAsInt();
    return v;
  }

  static SOCResourceSet resources(JsonArray a) {
    SOCResourceSet s = new SOCResourceSet();
    for (int i = 0; i < a.size(); i++) s.setAmount(a.get(i).getAsInt(), i + 1);
    return s;
  }

  static int[] counts(SOCResourceSet s) {
    int[] a = new int[5];
    for (int i = 0; i < 5; i++) a[i] = s.getAmount(i + 1);
    return a;
  }

  static int[] constant(String name) throws Exception {
    java.lang.reflect.Field f = SOCBoard4p.class.getDeclaredField(name);
    f.setAccessible(true);
    return (int[]) f.get(null);
  }

  static JsonObject geometry() throws Exception {
    SOCBoard b = new SOCBoard4p(null);
    JsonObject g = new JsonObject();
    JsonArray hexes = new JsonArray();
    TreeMap<Integer, int[]> edges = new TreeMap<>();
    for (int h : b.getLandHexCoords()) {
      JsonObject hex = new JsonObject();
      hex.addProperty("coord", h);
      int[] nodes = b.getAdjacentNodesToHex_arr(h);
      hex.add("nodes", JSON.toJsonTree(nodes));
      hexes.add(hex);
      for (int n : nodes)
        for (int e : b.getAdjacentEdgesToNode_arr(n))
          if (e != -9) {
            int[] ns = b.getAdjacentNodesToEdge_arr(e);
            if (b.isNodeOnLand(ns[0]) && b.isNodeOnLand(ns[1])) edges.put(e, ns);
          }
    }
    JsonArray es = new JsonArray();
    for (Map.Entry<Integer, int[]> e : edges.entrySet()) {
      JsonObject v = new JsonObject();
      v.addProperty("coord", e.getKey());
      v.add("nodes", JSON.toJsonTree(e.getValue()));
      es.add(v);
    }
    g.add("hexes", hexes);
    g.add("edges", es);
    g.add("port_edges", JSON.toJsonTree(b.getPortsEdges()));
    return g;
  }

  static final class Board extends SOCBoard4p {
    int[] robberChoices;

    Board(SOCGameOptionSet opts) {
      super(opts);
    }

    @Override
    public int[] getLandHexCoords() {
      return robberChoices == null ? super.getLandHexCoords() : robberChoices;
    }
  }

  static final class Opening extends OpeningBuildStrategy {
    Opening(SOCGame g, SOCPlayer p, SOCRobotBrain b) {
      super(g, p, b);
    }

    void restore() {
      if (!ourPlayerData.getSettlements().isEmpty())
        firstSettlement = ourPlayerData.getSettlements().firstElement().getCoordinates();
    }
  }

  static final class Planner extends SOCRobotDM {
    Planner(SOCRobotBrain b) {
      super(b);
    }

    SOCPossibleRoad freeRoad() {
      planStuff(SMART_STRATEGY);
      return favoriteRoad;
    }
  }

  static final class Client extends SOCRobotClient {
    JsonObject command;
    SOCPlayingPiece placedPiece;

    Client(String name) {
      super(new ServerConnectInfo("localhost", 8880, null), name, null);
    }

    void emit(String kind, int coord) {
      command = new JsonObject();
      command.addProperty("kind", kind);
      command.addProperty("coord", coord);
    }

    @Override
    public synchronized boolean put(String ignored) {
      return true;
    }

    @Override
    public void putPiece(SOCGame g, SOCPlayingPiece p) {
      placedPiece = p;
      emit(
          p.getType() == 0 ? "BUILD_ROAD" : p.getType() == 1 ? "BUILD_SETTLEMENT" : "BUILD_CITY",
          p.getCoordinates());
    }

    @Override
    public void buildRequest(SOCGame g, int type) {
      emit("REQUEST_BUILD", type);
    }

    @Override
    public void buyDevCard(SOCGame g) {
      emit("BUY_DEVELOPMENT_CARD", 0);
    }

    @Override
    public void rollDice(SOCGame g) {
      emit("ROLL", 0);
    }

    @Override
    public void endTurn(SOCGame g) {
      emit("END_TURN", 0);
    }

    @Override
    public void bankTrade(SOCGame g, SOCResourceSet give, SOCResourceSet get) {
      emit("MARITIME_TRADE", 0);
      command.add("give", JSON.toJsonTree(counts(give)));
      command.add("get", JSON.toJsonTree(counts(get)));
    }

    @Override
    public void playDevCard(SOCGame g, int type) {
      emit(
          type == SOCDevCardConstants.KNIGHT
              ? "PLAY_KNIGHT_CARD"
              : type == SOCDevCardConstants.ROADS
                  ? "PLAY_ROAD_BUILDING"
                  : type == SOCDevCardConstants.DISC ? "PLAY_YEAR_OF_PLENTY" : "PLAY_MONOPOLY",
          0);
    }

    @Override
    public void moveRobber(SOCGame g, SOCPlayer p, int hex) {
      emit("MOVE_ROBBER", hex);
    }

    @Override
    public void discard(SOCGame g, SOCResourceSet s) {
      emit("DISCARD_RESOURCE", 0);
      command.add("resources", JSON.toJsonTree(counts(s)));
    }
  }

  static final class Brain extends SOCRobotBrain {
    final Client output;

    Brain(Client c, SOCGame g) {
      super(
          c,
          new SOCRobotParameters(120, 35, .13f, 1f, 1f, 3f, 1f, SOCRobotDM.SMART_STRATEGY, 0),
          g,
          new CappedQueue<SOCMessage>());
      output = c;
      setOurPlayerData();
    }

    @Override
    public void pause(int ms) {}

    @Override
    protected SOCRobotDM createDM() {
      return new Planner(this);
    }

    @Override
    protected void setStrategyFields() {
      super.setStrategyFields();
      openingBuildStrategy = new Opening(game, ourPlayerData, this);
    }

    void acknowledge() {
      waitingForGameState = false;
      waitingForOurTurn = false;
      ourTurn = true;
      waitingForTradeMsg = false;
      waitingForTradeResponse = false;
      waitingForDevCard = false;
      expectPLACING_ROAD =
          expectPLACING_SETTLEMENT = expectPLACING_CITY = expectPLACING_SHIP = false;
      expectPLACING_ROBBER = expectWAITING_FOR_DISCOVERY = expectWAITING_FOR_MONOPOLY = false;
      expectPLACING_FREE_ROAD1 = expectPLACING_FREE_ROAD2 = false;
      expectPUTPIECE_FROM_START1A =
          expectPUTPIECE_FROM_START1B =
              expectPUTPIECE_FROM_START2A = expectPUTPIECE_FROM_START2B = false;
      expectDICERESULT = expectPLAY1 = expectDISCARD = false;
      counter = 0;
      doneTrading = true;
    }

    void newTurn() {
      resetFieldsAndBuildingPlan();
      failedBuildingAttempts = 0;
      failedBankTrades = 0;
      rejectedPlayDevCardType = -1;
    }

    JsonObject act(String phase, JsonArray legal, int discardCount) {
      output.command = null;
      output.placedPiece = null;
      acknowledge();
      ((Opening) openingBuildStrategy).restore();
      if (phase.equals("initial_settlement")) {
        HashSet<Integer> allowed = new HashSet<>();
        for (JsonElement x : legal) allowed.add(integer(x.getAsJsonObject(), "coord"));
        ourPlayerData.setPotentialAndLegalSettlements(allowed, false, null);
        if (ourPlayerData.getSettlements().isEmpty()) {
          game.setGameState(SOCGame.START1A);
          placeFirstSettlement(openingBuildStrategy.planInitialSettlements());
        } else {
          game.setGameState(SOCGame.START2A);
          placeInitSettlement(openingBuildStrategy.planSecondSettlement());
        }
      } else if (phase.equals("initial_road")) {
        game.setGameState(
            ourPlayerData.getRoadsAndShips().isEmpty() ? SOCGame.START1B : SOCGame.START2B);
        planAndPlaceInitRoad();
      } else if (phase.equals("discard")) {
        game.setGameState(SOCGame.WAITING_FOR_DISCARDS);
        discard(discardCount);
      } else if (phase.equals("robber")) {
        game.setGameState(SOCGame.PLACING_ROBBER);
        TreeSet<Integer> choices = new TreeSet<>();
        for (JsonElement x : legal) choices.add(integer(x.getAsJsonObject(), "coord"));
        Board board = (Board) game.getBoard();
        board.robberChoices = new int[choices.size()];
        int ci = 0;
        for (int h : choices) board.robberChoices[ci++] = h;
        try {
          moveRobber();
        } finally {
          board.robberChoices = null;
        }
        int hex = integer(output.command, "coord");
        boolean[] victims = new boolean[game.maxPlayers];
        for (JsonElement x : legal) {
          JsonObject a = x.getAsJsonObject();
          if (integer(a, "coord") == hex && integer(a, "victim") >= 0)
            victims[integer(a, "victim")] = true;
        }
        output.command.addProperty("victim", robberStrategy.chooseRobberVictim(victims, false));
      } else if (phase.equals("free_road")) {
        // Native Road Building plan was consumed once when playing the card.
        game.setGameState(SOCGame.PLACING_FREE_ROAD1);
        if (whatWeWantToBuild != null
            && ourPlayerData.isPotentialRoad(whatWeWantToBuild.getCoordinates()))
          output.putPiece(game, whatWeWantToBuild);
        else {
          SOCPossiblePiece p = null;
          if (!buildingPlan.isEmpty() && buildingPlan.getFirstPiece() instanceof SOCPossibleRoad)
            p = buildingPlan.advancePlan();
          if (p == null) p = ((Planner) decisionMaker).freeRoad();
          if (p != null)
            output.putPiece(game, new SOCRoad(ourPlayerData, p.getCoordinates(), null));
        }
      } else if (phase.equals("roll")) {
        game.setGameState(SOCGame.ROLL_OR_CARD);
        rollOrPlayKnightOrExpectDice();
      } else {
        game.setGameState(SOCGame.PLAY1);
        planAndDoActionForPLAY1();
        if (output.command != null
            && output.command.get("kind").getAsString().equals("REQUEST_BUILD"))
          output.putPiece(game, whatWeWantToBuild);
      }
      if (output.command == null)
        throw new IllegalStateException(
            "Native brain produced no command in "
                + phase
                + " plan="
                + buildingPlan
                + " piece="
                + whatWeWantToBuild
                + " failed="
                + failedBuildingAttempts);
      String kind = output.command.get("kind").getAsString();
      if (kind.equals("PLAY_YEAR_OF_PLENTY"))
        output.command.add(
            "resources", JSON.toJsonTree(counts(decisionMaker.getResourceChoices())));
      if (kind.equals("PLAY_MONOPOLY"))
        output.command.addProperty("resource", monopolyStrategy.getMonopolyChoice());
      return output.command;
    }

    void rejected(JsonObject cmd) {
      String kind = cmd.get("kind").getAsString();
      if (kind.startsWith("PLAY_")) {
        rejectedPlayDevCardType =
            kind.equals("PLAY_YEAR_OF_PLENTY")
                ? SOCDevCardConstants.DISC
                : kind.equals("PLAY_ROAD_BUILDING")
                    ? SOCDevCardConstants.ROADS
                    : kind.equals("PLAY_MONOPOLY")
                        ? SOCDevCardConstants.MONO
                        : SOCDevCardConstants.KNIGHT;
      } else if (kind.equals("MARITIME_TRADE")) {
        // Catanatron can reject a native trade when the finite bank is empty.
        // Feed the same rejection a real server sends back into the upstream
        // brain; its own retry limit and planning decide what to do next.
        handleREJECTOFFER(new SOCRejectOffer(game.getName(), -1, SOCRejectOffer.REASON_CANNOT_MAKE_TRADE));
      } else if (kind.startsWith("BUILD_")) {
        if (output.placedPiece != null) cancelWrongPiecePlacementLocal(output.placedPiece);
        resetBuildingPlan();
      } else throw new IllegalStateException("Native command has no legal translation: " + cmd);
    }
  }

  static final class Session {
    SOCGame game;
    Brain brain;
    String layoutKey;
    int eventCount = 0, freeRoads = 0;
    SOCResourceSet[] knowledge;
    int[] bank = {0, 19, 19, 19, 19, 19};
    int pieceCount = 0, index = -1, turn = -1, translationRejections = 0;
    String snapshot, cachedRequest;
    JsonArray appliedPieces = new JsonArray(), appliedEvents = new JsonArray();
    JsonObject cachedCommand;
    int cachedIndex = -1;

    Session(JsonObject o) throws Exception {
      SOCGameOptionSet known = SOCGameOptionSet.getAllKnownOptions(), opts = new SOCGameOptionSet();
      opts.setBoolOption("NT", known);
      opts.setIntOption("VP", integer(o, "vps"), true, known);
      layoutKey =
          o.get("hexes").toString()
              + o.get("ports")
              + o.get("players").getAsJsonArray().size()
              + o.get("vps");
      SOCGame.boardFactory = (options, large, max) -> new Board(options);
      game = new SOCGame("bridge", opts, known);
      SOCBoard b = game.getBoard();
      int[] hl = b.getHexLayout().clone(), nl = b.getNumberLayout().clone();
      Arrays.fill(hl, SOCBoard.WATER_HEX);
      Arrays.fill(nl, -1);
      for (JsonElement x : o.getAsJsonArray("hexes")) {
        int[] h = ints(x.getAsJsonArray());
        int n = b.getHexNumFromCoord(h[0]);
        hl[n] = h[1];
        nl[n] = h[2];
      }
      int[] portHex = constant("PORTS_HEXNUM_V1"), facing = b.getPortsFacing();
      int[] pts = ints(o.getAsJsonArray("ports"));
      for (int i = 0; i < pts.length; i++)
        hl[portHex[i]] = pts[i] == 0 ? 6 + facing[i] : (facing[i] << 4) + pts[i];
      b.setHexLayout(hl);
      b.setNumberLayout(nl);
      int n = o.getAsJsonArray("players").size();
      knowledge = new SOCResourceSet[n];
      for (int p = 0; p < n; p++) {
        knowledge[p] = new SOCResourceSet();
        game.addPlayer("seat" + p, p);
        game.getPlayer(p).setRobotFlag(true, false);
      }
      game.setFirstPlayer(0);
      game.setGameState(SOCGame.PLAY1);
      brain = new Brain(new Client("seat" + integer(o, "seat")), game);
    }

    void sync(JsonObject o) {
      JsonArray pieces = o.getAsJsonArray("pieces");
      game.setGameState(SOCGame.PLAY1);
      JsonArray events = o.getAsJsonArray("events");
      for (int i = eventCount; i < events.size(); i++) applyEvent(events.get(i).getAsJsonObject());
      eventCount = events.size();
      appliedEvents = events.deepCopy();
      pieceCount = pieces.size();
      appliedPieces = pieces.deepCopy();
      JsonArray players = o.getAsJsonArray("players");
      game.setPlayerWithLargestArmy(null);
      game.setPlayerWithLongestRoad(null);
      for (int i = 0; i < players.size(); i++) {
        JsonObject data = players.get(i).getAsJsonObject();
        SOCPlayer p = game.getPlayer(i);
        p.getResources().clear();
        SOCResourceSet observed = resources(data.getAsJsonArray("hand"));
        if (i == integer(o, "seat")) knowledge[i] = observed;
        else {
          // The native client retains publicly observed cards. A hidden loss
          // converts knowledge to UNKNOWN, exactly as the protocol client does.
          int total = observed.getTotal();
          if (knowledge[i].getKnownTotal() > total) knowledge[i].clear();
          knowledge[i].setAmount(
              total - knowledge[i].getKnownTotal(), SOCResourceConstants.UNKNOWN);
        }
        p.getResources().add(knowledge[i]);
        p.getInventory().clear();
        int[] types = {
          SOCDevCardConstants.KNIGHT,
          SOCDevCardConstants.MONO,
          SOCDevCardConstants.DISC,
          SOCDevCardConstants.ROADS,
          SOCDevCardConstants.CAP,
          SOCDevCardConstants.UNKNOWN
        };
        int[] old = ints(data.getAsJsonArray("old")), fresh = ints(data.getAsJsonArray("new"));
        for (int d = 0; d < types.length; d++) {
          if (old[d] > 0) p.getInventory().addDevCard(old[d], SOCInventory.OLD, types[d]);
          if (fresh[d] > 0) p.getInventory().addDevCard(fresh[d], SOCInventory.NEW, types[d]);
        }
        p.setPlayedDevCard(data.get("played").getAsBoolean());
        p.setNumKnights(integer(data, "knights"));
        p.setLongestRoadLength(integer(data, "longest"));
        if (data.get("army").getAsBoolean()) game.setPlayerWithLargestArmy(p);
        if (data.get("road").getAsBoolean()) game.setPlayerWithLongestRoad(p);
      }
      game.setCurrentPlayerNumber(integer(o, "current"));
      game.setRoundCount(integer(o, "round"));
      game.setNumDevCards(integer(o, "deck"));
      game.getBoard().setRobberHex(integer(o, "robber"), false);
      if (turn != integer(o, "turn")) {
        brain.newTurn();
        turn = integer(o, "turn");
      }
      game.setCurrentDice(integer(o, "dice"));
      String phase = o.get("phase").getAsString();
      game.setGameState(
          phase.equals("over")
              ? SOCGame.OVER
              : phase.equals("roll")
                  ? SOCGame.ROLL_OR_CARD
                  : phase.equals("robber")
                      ? SOCGame.PLACING_ROBBER
                      : phase.equals("discard")
                          ? SOCGame.WAITING_FOR_DISCARDS
                          : phase.equals("initial_settlement")
                              ? (game.getPlayer(integer(o, "current")).getSettlements().isEmpty()
                                  ? SOCGame.START1A
                                  : SOCGame.START2A)
                              : phase.equals("initial_road")
                                  ? (game.getPlayer(integer(o, "current"))
                                          .getRoadsAndShips()
                                          .isEmpty()
                                      ? SOCGame.START1B
                                      : SOCGame.START2B)
                                  : phase.equals("free_road")
                                      ? SOCGame.PLACING_FREE_ROAD1
                                      : SOCGame.PLAY1);
      index = integer(o, "index");
      snapshot = o.toString();
    }

    void transferBank(int seat, SOCResourceSet set, boolean gain) {
      for (int r = 1; r <= 5; r++) {
        int amount = set.getAmount(r);
        if (amount == 0) continue;
        if (gain) {
          knowledge[seat].add(amount, r);
          bank[r] -= amount;
        } else {
          knowledge[seat].subtract(amount, r);
          bank[r] += amount;
        }
      }
      if (knowledge[seat].getAmount(6) < 0) knowledge[seat].setAmount(0, 6);
    }

    void applyEvent(JsonObject e) {
      int pn = integer(e, "seat");
      String kind = e.get("kind").getAsString();
      SOCPlayer p = game.getPlayer(pn);
      boolean initial = e.get("initial").getAsBoolean();
      if (kind.startsWith("BUILD_")) {
        int type =
            kind.equals("BUILD_ROAD")
                ? SOCPlayingPiece.ROAD
                : kind.equals("BUILD_SETTLEMENT")
                    ? SOCPlayingPiece.SETTLEMENT
                    : SOCPlayingPiece.CITY;
        int coord = integer(e, "coord");
        SOCPlayingPiece piece =
            type == 0
                ? new SOCRoad(p, coord, null)
                : type == 1 ? new SOCSettlement(p, coord, null) : new SOCCity(p, coord, null);
        game.setCurrentPlayerNumber(pn);
        game.putPiece(piece);
        if (initial && type == 1 && p.getSettlements().size() == 2) {
          SOCResourceSet gained = new SOCResourceSet();
          for (int h : game.getBoard().getAdjacentHexesToNode(coord)) {
            int r = game.getBoard().getHexTypeFromCoord(h);
            if (r >= 1 && r <= 5) gained.add(1, r);
          }
          transferBank(pn, gained, true);
        }
        brain.handlePUTPIECE_updateTrackers(pn, coord, type);
        int roadCount = 0;
        for (int seat = 0; seat < knowledge.length; seat++)
          roadCount += game.getPlayer(seat).getRoadsAndShips().size();
        if (initial && type == 0 && roadCount == knowledge.length * 2) {
          game.setGameState(SOCGame.START2B);
          game.setGameState(SOCGame.ROLL_OR_CARD);
          game.setGameState(SOCGame.PLAY1);
        }
        if (!initial) {
          if (type == 0 && freeRoads > 0) freeRoads--;
          else transferBank(pn, SOCPlayingPiece.getResourcesToBuild(type), false);
        }
      } else if (kind.equals("ROLL")) {
        int dice = integer(e, "dice");
        game.setCurrentDice(dice);
        if (dice != 7) {
          SOCResourceSet[] gained = new SOCResourceSet[knowledge.length];
          int[] demand = new int[6];
          for (int i = 0; i < gained.length; i++) {
            gained[i] = game.getResourcesGainedFromRoll(game.getPlayer(i), dice);
            for (int r = 1; r <= 5; r++) demand[r] += gained[i].getAmount(r);
          }
          for (int i = 0; i < gained.length; i++) {
            for (int r = 1; r <= 5; r++) if (demand[r] > bank[r]) gained[i].setAmount(0, r);
          }
          for (int i = 0; i < gained.length; i++) transferBank(i, gained[i], true);
        }
      } else if (kind.equals("BUY_DEVELOPMENT_CARD")) transferBank(pn, SOCDevCard.COST, false);
      else if (kind.equals("MARITIME_TRADE")) {
        transferBank(pn, resources(e.getAsJsonArray("give")), false);
        transferBank(pn, resources(e.getAsJsonArray("get")), true);
      } else if (kind.equals("PLAY_YEAR_OF_PLENTY"))
        transferBank(pn, resources(e.getAsJsonArray("resources")), true);
      else if (kind.equals("DISCARD_RESOURCE")) {
        SOCResourceSet loss = new SOCResourceSet();
        loss.add(1, integer(e, "resource"));
        transferBank(pn, loss, false);
      } else if (kind.equals("PLAY_ROAD_BUILDING")) freeRoads = 2;
      else if (kind.equals("END_TURN")) freeRoads = 0;
      else if (kind.equals("MOVE_ROBBER")) {
        game.getBoard().setRobberHex(integer(e, "coord"), false);
        int victim = integer(e, "victim");
        if (victim >= 0) {
          int r = integer(e, "stolen");
          knowledge[victim].subtract(1, r, r == 6);
          knowledge[pn].add(1, r);
        }
      } else if (kind.equals("PLAY_MONOPOLY")) {
        int r = integer(e, "resource");
        for (int i = 0; i < knowledge.length; i++)
          if (i != pn) {
            int amount = knowledge[i].getAmount(r);
            knowledge[pn].add(amount, r);
            knowledge[i].setAmount(0, r);
          }
      }
    }

    JsonObject status() {
      JsonObject s = new JsonObject();
      s.addProperty("engine", "SOCRobotBrain");
      s.addProperty("strategy", "SMART");
      s.addProperty("translation_rejections", translationRejections);
      s.addProperty("game_state", game.getGameState());
      s.addProperty("state_index", index);
      s.addProperty("pieces", pieceCount);
      s.addProperty("robber", game.getBoard().getRobberHex());
      s.addProperty("current_seat", game.getCurrentPlayerNumber());
      s.addProperty("vps_to_win", game.vp_winner);
      JsonArray players = new JsonArray();
      for (int i = 0; i < game.maxPlayers; i++) {
        if (game.isSeatVacant(i)) continue;
        SOCPlayer p = game.getPlayer(i);
        JsonObject v = new JsonObject();
        v.add("hand", JSON.toJsonTree(p.getResources().getAmounts(true)));
        v.addProperty("settlements", p.getSettlements().size());
        v.addProperty("cities", p.getCities().size());
        v.addProperty("roads", p.getRoadsAndShips().size());
        boolean[] ports = new boolean[6];
        for (int r = 0; r < 6; r++) ports[r] = p.getPortFlag(r);
        v.add("ports", JSON.toJsonTree(ports));
        players.add(v);
      }
      s.add("players", players);
      JsonArray hexes = new JsonArray();
      for (int h : game.getBoard().getLandHexCoords())
        hexes.add(
            JSON.toJsonTree(
                new int[] {
                  h,
                  game.getBoard().getHexTypeFromCoord(h),
                  game.getBoard().getNumberOnHexFromCoord(h)
                }));
      s.add("hexes", hexes);
      return s;
    }
  }

  static boolean matches(JsonObject cmd, JsonObject legal) {
    String k = cmd.get("kind").getAsString();
    if (!k.equals(legal.get("kind").getAsString())) return false;
    if (k.startsWith("BUILD_") || k.equals("MOVE_ROBBER"))
      if (integer(cmd, "coord") != integer(legal, "coord")) return false;
    if (k.equals("MOVE_ROBBER")) return integer(cmd, "victim") == integer(legal, "victim");
    if (k.equals("MARITIME_TRADE"))
      return cmd.get("give").equals(legal.get("give")) && cmd.get("get").equals(legal.get("get"));
    if (k.equals("PLAY_YEAR_OF_PLENTY")) return cmd.get("resources").equals(legal.get("resources"));
    if (k.equals("PLAY_MONOPOLY")) return integer(cmd, "resource") == integer(legal, "resource");
    if (k.equals("DISCARD_RESOURCE"))
      return cmd.getAsJsonArray("resources").get(integer(legal, "resource") - 1).getAsInt() > 0;
    return true;
  }

  public static void main(String[] args) throws Exception {
    if (args.length > 0 && args[0].equals("--health")) {
      System.out.println("JSettlers 2.7.00 SOCRobotBrain adapter v2");
      return;
    }
    // Upstream diagnostics must never corrupt the JSON wire stream.
    PrintStream wire = System.out;
    System.setOut(System.err);
    BufferedReader input = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
    String line;
    while ((line = input.readLine()) != null) {
      JsonObject reply = new JsonObject();
      try {
        JsonObject req = new JsonParser().parse(line).getAsJsonObject();
        String op = req.get("op").getAsString();
        if (op.equals("geometry")) reply = geometry();
        else {
          String key = req.get("session").getAsString();
          JsonObject o = req.getAsJsonObject("state");
          Session s = sessions.get(key);
          if (o != null) {
            boolean rebuild = s == null;
            if (s != null) {
              String layout =
                  o.get("hexes").toString()
                      + o.get("ports")
                      + o.get("players").getAsJsonArray().size()
                      + o.get("vps");
              rebuild =
                  integer(o, "index") < s.index
                      || !layout.equals(s.layoutKey)
                      || o.getAsJsonArray("pieces").size() < s.pieceCount;
              if (!rebuild)
                for (int i = 0; i < s.eventCount; i++)
                  if (i >= o.getAsJsonArray("events").size()
                      || !s.appliedEvents.get(i).equals(o.getAsJsonArray("events").get(i))) {
                    rebuild = true;
                    break;
                  }
            }
            if (rebuild) {
              s = new Session(o);
              sessions.put(key, s);
            }
            s.sync(o);
          }
          if (s == null) throw new IllegalArgumentException("Unknown session");
          reply = s.status();
          if (op.equals("decide")) {
            String fingerprint = o.toString() + req.get("phase") + req.get("legal");
            if (fingerprint.equals(s.cachedRequest) && s.cachedCommand != null)
              reply = s.cachedCommand.deepCopy();
            else {
              JsonArray legal = req.getAsJsonArray("legal");
              boolean found = false;
              for (int attempt = 0; attempt < 12 && !found; attempt++) {
                JsonObject cmd =
                    s.brain.act(
                        req.get("phase").getAsString(), legal, integer(req, "discard_count"));
                for (int i = 0; i < legal.size(); i++)
                  if (matches(cmd, legal.get(i).getAsJsonObject())) {
                    reply.addProperty("action_index", i);
                    reply.add("command", cmd);
                    found = true;
                    break;
                  }
                if (!found) {
                  s.translationRejections++;
                  s.brain.rejected(cmd);
                }
              }
              if (!found)
                throw new IllegalStateException("Native brain could not produce a legal command");
              s.cachedIndex = s.index;
              s.cachedRequest = fingerprint;
              s.cachedCommand = reply.deepCopy();
            }
          }
          while (sessions.size() > 128) sessions.remove(sessions.keySet().iterator().next());
        }
      } catch (Throwable e) {
        e.printStackTrace(System.err);
        reply = new JsonObject();
        reply.addProperty("error", e.toString());
      }
      wire.println(JSON.toJson(reply));
      wire.flush();
    }
  }
}
