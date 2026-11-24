#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p integrations/jsettlers/build/classes integrations/jsettlers/lib
find integrations/jsettlers/upstream/src/main/java -name '*.java' > integrations/jsettlers/build/sources.txt
javac -encoding UTF-8 -cp integrations/jsettlers/lib/gson-2.8.6.jar -d integrations/jsettlers/build/classes @integrations/jsettlers/build/sources.txt agent_apis/jsettlers/Bridge.java 2> integrations/jsettlers/build/compiler.log
jar cf integrations/jsettlers/lib/jsettlers-adapter.jar -C integrations/jsettlers/build/classes . -C integrations/jsettlers/upstream/src/main/resources . -C integrations/jsettlers/upstream COPYING-GPLv3.txt
java -cp 'integrations/jsettlers/lib/*' Bridge --health
