(ns crypto-market-snap.core
  (:require
    [compojure.core :as c]
    [compojure.handler :as h]
    [ring.adapter.jetty :as jetty]
    [environ.core :as e]
    [clj-http.client :as http]
    [clojure.string :as str]
    [clojure.edn :as edn]
    [clojure.java.io :as io]
    [clojure.tools.logging :as log])
  (:import
    [org.quartz.impl StdSchedulerFactory]
    [org.quartz Scheduler JobDetail TriggerBuilder JobBuilder CronScheduleBuilder Job CronTrigger]
    (java.io File))
  (:gen-class))

(def markets
  ["luno:btczar"
   "kraken:btcusd"
   "kraken:ethusd"
   "poloniex:btcusd"
   "poloniex:ethusd"
   "bitfinex:btcusd"
   "bitfinex:ethusd"
   "quoine:btcusd"
   "quoine:ethusd"])

(defonce usage-stats
  (atom {}))

(defonce state
  (atom (try (edn/read-string (slurp "state.edn")) (catch Exception e {}))))

(defn swap-state [f & args]
  (let [r (apply swap! state f args)]
    (spit "state.edn" (pr-str r))
    r))

(defn api-call [path usage-tag & [opts]]
  (let [{:keys [body status] :as r}
        (http/get (str "https://api.cryptowat.ch" path)
          (merge opts {:as :json}))]
    (if (= 200 status)
      (do
        (when (:allowance body)
          (log/info usage-tag ", API Call: " path " - Cost: " (:cost (:allowance body)) " -- " (:remaining (:allowance body)) " left")
          (swap! usage-stats
            (fn [s]
              (-> s
                  (update usage-tag (fnil + 0) (get-in body [:allowance :cost] 0))
                  (assoc :remaining (get-in body [:allowance :remaining]))))))
        (:result body))
      (log/error "Invalid response" r))))

(defn pathify [market]
  (str "/markets/" (str/replace market ":" "/")))

(defn write-response [market action data]
  (let [parent (io/file (str "data/" market "/" action))]
    (.mkdirs parent)
    (spit (io/file parent (str market ":" action ":" (System/currentTimeMillis) ".edn"))
      (pr-str data))))

(defn trades [market]
  (let [last-trade (get-in @state [market :last-trade])
        result
        (api-call (str (pathify market) "/trades") market
          (when last-trade
            {:query-params {:since last-trade}}))
        _ (write-response market "trades" result)

        [[_ last-tx _ _] :as txs]
        (reverse (sort-by (fn [[id dt price vol]] dt) result))]
    (swap-state assoc-in [market :last-trade] last-tx)
    result))

(defn pull-trades []
  (doseq [m markets]
    (future
      (try
        (trades m)
        (catch Exception e
          (.printStackTrace e))))))

(defn price [market]
  (let [result
        (api-call (str (pathify market) "/price") market)
        _ (write-response market "price" result)]
    (swap-state assoc-in [market :last-price] (:price result))
    result))

(defn pull-prices []
  (try
    (let [result
          (api-call "/markets/prices" "prices")
          _ (write-response "markets" "price" result)]
      result)
    (catch Exception e
      (.printStackTrace e))))

(defn orderbook [market]
  (let [result
        (api-call (str (pathify market) "/orderbook") market)
        _ (write-response market "orderbook" result)]
    (swap-state assoc-in [market :orders] {:asks (count (:asks result)) :bids (count (:bids result))})
    result))

(defn pull-data []
  (doseq [m markets]
    (future
      (try
        (orderbook m)
        (catch Exception e
          (.printStackTrace e))))))





(defn index [req]
  {:status 200
   :headers {"Content-Type" "text/html"}
   :body
   (str "<html><body><p><b>Usage:</b><br/>" (pr-str @usage-stats)
     "</p><p><b>State:</b><br/> " (pr-str @state) "</p></body></html>")})

(c/defroutes app
  (c/GET "/" [] index))



(def cronargs (atom {}))

(deftype ImportJob []
  Job
  (execute [_ context]
    (let [jobkey (.getKey ^JobDetail (.getJobDetail context))
          [fn args] (get @cronargs jobkey)]
      (apply fn args))))

(defn get-scheduler []
  (StdSchedulerFactory/getDefaultScheduler))

(defn schedule-job [^Scheduler scheduler cron fn & args]
  (let [trigger (TriggerBuilder/newTrigger)
        trigger (-> trigger
                    ;(withIdentity "import-trigger" "import")
                    (.withSchedule (CronScheduleBuilder/cronSchedule ^String cron))
                    (.build))
        job     (JobBuilder/newJob ImportJob)
        job     ^JobDetail (-> job #_(.withIdentity "name" "group") .build)
        jobkey (.getKey job)]
    (swap! cronargs assoc jobkey [fn args])
    (.scheduleJob scheduler job trigger)
    (.start scheduler)
    (.getKey job)))

(defn stop-job [^Scheduler scheduler jobkey]
  (.deleteJob scheduler jobkey)
  (swap! cronargs dissoc jobkey))

(defn stop-scheduler [^Scheduler scheduler & [wait?]]
  (if wait?
    (.shutdown scheduler true)
    (.shutdown scheduler)))

(defn print-state []
  (log/info "Usage: " (pr-str @usage-stats))
  (log/info "State: " (pr-str @state)))

(defn read-orderbook [file]
  (let [book (edn/read-string (slurp file))
        [_ dt] (re-find #"[:]([0-9]+)[.]edn$" (.getName file))]
    (assoc book :date (Long/parseLong dt))))

(defn mean [coll]
  (let [sum (apply + coll)
        count (count coll)]
    (if (pos? count)
      (/ sum count)
      0)))

(defn standard-deviation [coll]
  (let [avg (mean coll)
        squares (for [x coll]
                  (let [x-avg (- x avg)]
                    (* x-avg x-avg)))
        total (count coll)]
    (-> (/ (apply + squares)
          (- total 1))
        (Math/sqrt))))

(defn aggregate-book [{:keys [asks bids date] :as book} value-percentage pips]
  (let [middle-max (ffirst (sort-by first asks))
        middle-min (ffirst (reverse (sort-by first bids)))
        center (/ (+ middle-min middle-max) 2)
        width (int (* center value-percentage))
        bucketed
        (->>
          (concat bids asks)
          (filter (fn [[p v]] (< (- center width) p (+ center width))))
          (map (fn [[p v]] [(int (* pips (/ (- p (- center width)) (* width 2)))) v]))
          (group-by first)
          (map (fn [[k g]] [k (reduce + (map second g))]))
          (into {}))
        result
        (->>
          (range pips)
          (map (fn [x] (get bucketed x 0)) )
          vec)
        value-cap (+ (mean result) (standard-deviation result))
        result-normalized
        (map
          (fn [v]
            (min (/ v value-cap) 1.0))
          result)
        ]
    {:date date
     :volume (reduce + (map second bucketed))
     :buckets result-normalized
     :price (float center)}))

(defn label-aggregates [book min1 min15 min60]
  (let [p (:price book)
        [p1t p15t p60t] [0.00225 0.03 0.06]
        p1 (* 10 (/ (- (:price min1) p) p))
        p15 (* 10 (/ (- (:price min15) p) p))
        p60 (* 10 (/ (- (:price min60) p) p))]
    (assoc book
      :labels
      [(if (> p1 p1t) 1 0)
       (if (< p1 (- p1t)) 1 0)
       (if (> p1t p1 (- p1t)) 1 0)
       (if (> p15 p15t) 1 0)
       (if (< p15 (- p15t)) 1 0)
       (if (> p15t p15 (- p15t)) 1 0)
       (if (> p60 p60t) 1 0)
       (if (< p60 (- p60t)) 1 0)
       (if (> p60t p60 (- p60t)) 1 0)])))

(defn visual-aggregate [{:keys [date buckets price labels] :as book}]
  (try
    (let [chars [" " "." "," "-" "+" "=" ":" "|" "#" "X" "^"]]
      (str
        (apply str
          date ": " (format "%.2f" (float price)) ": "
          (map (fn [v] (get chars (int (* 10 v)))) buckets))
        " " (pr-str labels)))
    (catch Exception e
      (.printStackTrace e)
      (log/warn (pr-str book)))))

(defn aggregate-orderbook [market]
  (let [books
        (->>
          (.listFiles (io/file (str "data/" market "/orderbook")))
          seq
          (pmap read-orderbook)
          (remove nil?)
          (sort-by :date)
          (pmap #(aggregate-book % 0.05 40)))
        labelled-books
        (map label-aggregates books (drop 15 books) (drop 60 books) (drop 120 books))
        parent (io/file (str "data/aggregates"))
        _ (.mkdirs parent)
        fname (str parent "/" market ":aggregate:" (:date (first books)) "-" (:date (last books)) ".edn")]
    (spit fname (pr-str labelled-books))
    #_(log/info market " aggregated to " fname)
    #_(doseq [b labelled-books] (-> b visual-aggregate log/info))))

(defn aggregate-orderbooks []
  (doseq [m markets]
    (aggregate-orderbook m)))

(defn -main [& args]
  (if (= (first args) "aggregate")
    (do
      (aggregate-orderbooks)
      (log/info "All aggregated")
      (println "done"))

    (do
      ;; Setup Quartz
      (schedule-job (get-scheduler) "0 */5 * * * ?" #'pull-trades)
      (schedule-job (get-scheduler) "0 */1 * * * ?" #'pull-data)
      (schedule-job (get-scheduler) "0 */1 * * * ?" #'pull-prices)


      (schedule-job (get-scheduler) "0 * * * * ?" #'print-state)
      ;; Setup Ring
      (jetty/run-jetty (h/site #'app) {:port (Integer/parseInt (str (or (first args) (e/env :port) 5000)))}))))
