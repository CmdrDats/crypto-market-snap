(ns crypto-market-snap.learning
  (:require [clojure.java.io :as io]
            [clojure.edn :as edn])
  (:import
    (java.util LinkedList ArrayList Vector Iterator Random Date)
    (java.io File)
    (org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder MultiLayerConfiguration Updater GradientNormalization BackpropType)
    (org.deeplearning4j.nn.api OptimizationAlgorithm Layer)
    (org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer$Builder ConvolutionLayer ConvolutionLayer$Builder SubsamplingLayer SubsamplingLayer$Builder SubsamplingLayer$PoolingType DenseLayer RBM$Builder RBM$VisibleUnit RBM$HiddenUnit RnnOutputLayer$Builder GravesLSTM$Builder)
    (org.deeplearning4j.nn.weights WeightInit)
    (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
    (org.nd4j.linalg.lossfunctions LossFunctions$LossFunction)
    (org.deeplearning4j.optimize.listeners ScoreIterationListener)
    (org.nd4j.linalg.dataset DataSet)
    (org.deeplearning4j.eval Evaluation)
    (org.deeplearning4j.util ModelSerializer)
    (org.nd4j.linalg.convolution Convolution$Type)
    (java.awt.image BufferedImage WritableRaster)
    (javax.imageio ImageIO)
    (org.nd4j.linalg.activations Activation)
    (org.nd4j.linalg.dataset.api.iterator DataSetIterator)
    (org.nd4j.linalg.factory Nd4j)
    (org.nd4j.linalg.api.ndarray INDArray))
  )

(def hyper-params
  {:initial
   {:orderbook-width 40
    :end-states 9
    :seed (rand)
    :iterations 1
    :learning-rate 0.006
    :rms-decay 0.95
    :backprop-forward-length 50
    :backprop-backward-length 100}})



(defn load-net [nm]
  {:name nm
   :model
   (ModelSerializer/restoreMultiLayerNetwork (File. (str nm ".net")))})

(defn save-net [{^MultiLayerNetwork model :model nm :name}]
  (ModelSerializer/writeModel model (File. (str nm ".net")) true)
  (ModelSerializer/writeModel model (File. (str nm ".cnet")) false))

(def optimizations
  {:sgd OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
   :line OptimizationAlgorithm/LINE_GRADIENT_DESCENT
   :hessian OptimizationAlgorithm/HESSIAN_FREE
   :conjugate OptimizationAlgorithm/CONJUGATE_GRADIENT
   :lbfgs OptimizationAlgorithm/LBFGS})

(def updaters
  {:nesterovs Updater/NESTEROVS
   })


(defn initial-config [hyper-params]
  {:name "initial"
   :config
   (->
     (NeuralNetConfiguration$Builder.)
     (.seed (int (:seed hyper-params)))
     (.iterations (:iterations hyper-params))
     (.learningRate (:learning-rate hyper-params))
     (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
     (.weightInit WeightInit/XAVIER)
     (.updater Updater/RMSPROP)
     (.rmsDecay (:rms-decay hyper-params))
     (.regularization true)
     (.l2 1e-4)
     (.list)
     (.layer 0
       (-> (GravesLSTM$Builder.)
           (.nIn (:orderbook-width hyper-params))
           (.nOut 100)
           (.activation Activation/TANH)
           (.gateActivationFunction Activation/HARDSIGMOID)
           (.dropOut 0.1)
           (.build)))
     (.layer 1
       (-> (GravesLSTM$Builder.)
           (.nIn 100)
           (.nOut 50)
           (.activation Activation/TANH)
           (.gateActivationFunction Activation/HARDSIGMOID)
           (.dropOut 0.1)
           (.build)))
     (.layer 2
       (-> (DenseLayer$Builder.)
           (.nIn 50)
           (.nOut 100)
           (.activation Activation/RELU)
           (.build)))
     (.layer 3
       (-> (RnnOutputLayer$Builder. LossFunctions$LossFunction/MSE)
           (.nIn 100)
           (.nOut (:end-states hyper-params))
           (.activation Activation/IDENTITY)
           (.weightInit WeightInit/XAVIER)
           (.build)))
     (.backpropType BackpropType/TruncatedBPTT)
     (.tBPTTForwardLength (:backprop-forward-length hyper-params))
     (.tBPTTForwardLength (:backprop-backward-length hyper-params))
     (.pretrain false)
     (.backprop true))})



(defn dataset-iterator [data minibatch-size example-length]
  (let [batches
        (->>
          (partition (int (/ example-length 2)) data)
          (reductions
            (fn [a b] [(second a) b]) [])
          (drop 2)
          (map #(apply concat %)))
        mini-batches (partition-all minibatch-size batches)
        initial-state {:batch-idx -1}
        state (atom initial-state)]
    ;; TODO: This is stupid - can do without using DataSetIterator.
    ;; Just need to end up with a way to generate DataSet instances on the fly. No big deal.
    (proxy [DataSetIterator] []
      (next []
        (let [batch-idx (:batch-idx (swap! state update :batch-idx inc))
              actual-size (count mini-batches)
              _ (println "next: " batch-idx "/" actual-size)
              mini-batch (nth mini-batches batch-idx)

              input (Nd4j/create (int-array [(count mini-batch) (count (:buckets (first data))) (count (first mini-batch))]) \f)
              output (Nd4j/create (int-array [(count mini-batch) (count (:labels (first data))) (count (first mini-batch))]) \f)]
          (doseq [bidx (range (count mini-batch))
                  :let [batch (nth mini-batch bidx)]]
            (doseq [c (range (count batch))
                    :let [entry (nth batch c)]]
              (doseq [e (range (count (:buckets entry)))]
                (.putScalar input ^ints (int-array (seq [bidx e c])) (double (nth (:buckets entry) e))))
              (doseq [e (range (count (:labels entry)))]
                (.putScalar output ^ints (int-array (seq [bidx e c])) (double (nth (:labels entry) e))))))
          (DataSet. input output)))

      (totalExamples []
        (count batches))
      (inputColumns []
        (count (:buckets (first data))))
      (totalOutcomes []
        (count (:labels (first data))))
      (resetSupported [] true)
      (asyncSupported [] false)
      (reset []
        (reset! state initial-state))
      (cursor []
        (:batch-idx @state))
      (numExamples []
        (count batches))
      (setPreProcessor []
        (UnsupportedOperationException. "Not Implemented"))
      (getPreProcessor []
        (UnsupportedOperationException. "Not Implemented"))
      (getLabels []
        ["15Min Buy" "15Min Sell" "15Min Stay"
         "60Min Buy" "60Min Sell" "60Min Stay"
         "120Min Buy" "120Min Sell" "120Min Stay"])
      (hasNext []
        (< (:batch-idx @state) (dec (count mini-batches)))))))

(defn load-dataset [f]
  (println "Loading " f)
  (let [data (edn/read-string (slurp f))]
    (dataset-iterator data 6 240)))

(def train-markets #{"bitfinex:btcusd" "kraken:btcusd" "luno:btczar" "poloniex:btcusd" "quoine:btcusd"})
#_(def train-markets #{"bitfinex:btcusd" "kraken:btcusd" "luno:btczar" "poloniex:btcusd" "quoine:btcusd"})
(def test-markets #{"bitfinex:ethusd" "kraken:ethusd" "poloniex:ethusd" "quoine:ethusd"})



(defn load-sets [market-set]
  (let [find-market #(second (re-find #"([^:]+[:][^:]+)[:]" (.getName %)))]
    (->> (.listFiles (io/file "data/aggregates"))
      seq
      (filter (comp market-set find-market))
      (map load-dataset))))

(defn evaluate-training [modelconf testset]
  (println "Evaluate on Training set:")
  (.reset testset)

  (doseq [test (iterator-seq testset)]
    (let [out (.output (:model modelconf) (.getFeatures test))
          labels (.getLabels test)]


      (doseq [c (range 120)]
        (print "[ ")
        (doseq [l (range 9)]
          (print (if (> (.getFloat ^INDArray labels (int-array [0 l c])) 0) "X" ".")))
        (print " ] ")
        (print " => [ ")
        (doseq [l (range 9)]
          (print
            (cond
              (> (.getFloat ^INDArray out (int-array [0 l c])) 0.9) "X"
              (> (.getFloat ^INDArray out (int-array [0 l c])) 0.5) "-"
              :else ".")))
        (print " ] ")

        (println ""))

      )

    )

  #_(let [eval (.evaluate (:model modelconf) testset)]
    (println (.stats eval))
    #_(println
      (pr-str
        {:time (Date.)
         :score (.score (:model modelconf))
         :accuracy (.accuracy eval)
         :precision (.precision eval)
         :recall (.recall eval)
         :f1 (.f1 eval)}))))

(defn fit [modelconf trainsets testsets epochs]
  (doseq [e (range epochs)
          dataset trainsets]
    (println "Epoch: " e)
    (.reset dataset)
    (while (.hasNext dataset)
      (.fit (:model modelconf) (.next dataset))
      (.rnnClearPreviousState (:model modelconf))
      )
    )

  (println "Evaluating against Training: ")
  (doseq [dataset trainsets]
    (evaluate-training modelconf dataset))

  (println "Evaluating Test: ")
  (doseq [dataset testsets]
    (evaluate-training modelconf dataset))
  (println "Done."))



(defonce net (atom nil))


(defn setup [conf-fn hyper-param-key]
  (let [conf (conf-fn (get hyper-params hyper-param-key))
        ^MultiLayerNetwork model (MultiLayerNetwork. (.build (:config conf)))]
    (.init model)
    #_(.setListeners model [(HistogramIterationListener. 1)])
    (.setListeners model [(ScoreIterationListener. 1)])
    (spit (str (:name conf) ".log") "")
    (reset! net (assoc conf :model model))))

(comment

  (defn evaluate [modelconf]
    (let [test (testset)]
      (.reset test)
      (println "Evaluate:")
      (let [eval (Evaluation. 3)]
        (doseq [^DataSet test (iterator-seq test)]
          (.eval eval (.getLabels test) (.output (:model modelconf) (.getFeatureMatrix test))))

        (let [stats
              {:time (Date.)
               :score (.score (:model modelconf))
               :accuracy (.accuracy eval)
               :precision (.precision eval)
               :recall (.recall eval)
               :f1 (.f1 eval)}]
          (spit (str (:name modelconf) ".log")
            (str
              (pr-str stats) "\r\n")
            :append true)

          (println (.stats eval))
          stats))))

  (defn evaluate-training [modelconf]
    (let [test (trainset)]
      (.reset test)
      (println "Evaluate on Training set:")
      (let [eval (Evaluation. 3)]
        (doseq [^DataSet test (iterator-seq test)]
          (.eval eval (.getLabels test) (.output (:model modelconf) (.getFeatureMatrix test))))

        (println (.stats eval))
        {:time (Date.)
         :score (.score (:model modelconf))
         :accuracy (.accuracy eval)
         :precision (.precision eval)
         :recall (.recall eval)
         :f1 (.f1 eval)})))




  (def stop-cycle (atom false))

  (defn fit-eval-cycle [modelconf]
    (doto
      (Thread.
        #(do
           (loop [score 0.0]
             (cond
               @stop-cycle true
               :else
               (do
                 (time (fit modelconf 1))
                 (let [stats (time (evaluate modelconf))]
                   (if (> (:f1 stats) score)
                     (do
                       (save-net modelconf)
                       (recur (:f1 stats)))
                     (recur score))))))
           (reset! stop-cycle false)))
      (.setDaemon true)
      (.start)))

  (defn stop-fit []
    (reset! stop-cycle true)))


