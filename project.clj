(defproject crypto-market-snap "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies
  [[org.clojure/clojure "1.8.0"]
   [compojure "1.5.2"]
   [ring/ring-jetty-adapter "1.6.0-beta6"]
   [ring/ring-core "1.6.1"]
   [environ "1.1.0"]
   [clj-http "3.4.1"]
   [cheshire "5.7.0"]
   [org.quartz-scheduler/quartz "2.2.3"]
   [ch.qos.logback/logback-classic "1.2.3"]
   [ch.qos.logback/logback-core "1.2.3"]
   [com.stuartsierra/log.dev "0.2.0"]
   [org.clojure/tools.logging "0.3.1"]

   [org.nd4j/nd4j "0.9.1" :extension "pom"]
   #_[org.nd4j/nd4j-cuda-7.5 "0.9.1"]
   [org.nd4j/nd4j-native "0.9.1"]
   [org.datavec/datavec-api "0.9.1"]
   [org.deeplearning4j/deeplearning4j-core "0.9.1"]
   ]
  :min-lein-version "2.0.0"
  :main crypto-market-snap.core)
