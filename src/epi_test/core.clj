(ns epi-test.core
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clj-boost.core :as boost]))

(defn transpose
  [m]
  (apply map list m))

(def input-results
  (->> (io/reader (io/resource "phenotype.csv"))
       csv/read-csv
       rest
       (map (juxt #(nth % 0) #(nth % 4)))
       (map (fn [[name phenotype]]
              [name (case phenotype
                      "never" 0
                      "former" 1
                      "current" 2
                      "NA" nil)]))
       (into {})))

(def input-data
  (->> (io/reader (io/resource "maas_cpgs.csv"))
       csv/read-csv
       transpose
       rest
       (map (partial mapv read-string))
       (map (fn [[gsm & others]]
              (when-let [phenotype (get input-results (name gsm))]
                (concat others [phenotype]))))
       (remove nil?)))

(defn train-test-split
  [dataset n]
  (let [shuffled (shuffle dataset)]
    (split-at n shuffled)))

(defn create-set
  [s]
  {:x (mapv drop-last s)
   :y (mapv last s)})

(defn train-model
  [training-set]
  (let [data (boost/dmatrix training-set)
        params {:params {:eta 0.00001
                         :objective "multi:softmax"
                         :num_class 3}
                :rounds 10
                :watches {:train data}
                :early-stopping 10}]
    (boost/fit data params)))

(defn predict-model
  [model testing-set]
  (boost/predict model (boost/dmatrix testing-set)))

(defn accuracy
  [predicted real]
  (let [right (map #(compare %1 %2) predicted real)]
    (float (/ (count (filter zero?  right))
              (count real)))))

(defn -main
  []
  (let [split-set    (train-test-split input-data (/ (count input-data) 2))
        [train test] (map create-set split-set)
        model        (train-model train)
        result       (predict-model model test)]
    ;; (println "Prediction:" (mapv int result))
    ;; (println "Real:      " (:y test))
    ;; (println "Accuracy:  " (accuracy result (:y test)))
    (accuracy result (:y test))))
