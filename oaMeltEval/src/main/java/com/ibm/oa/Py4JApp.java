package com.ibm.oa;

import py4j.GatewayServer;

public class Py4JApp {
    /*
     * This class can be called from python via Py4J. see https://www.py4j.org/
     */
    public static void main(String[] args) throws Exception {
        MeltWebApiEvaluator evaluator = new MeltWebApiEvaluator();
        GatewayServer server = new GatewayServer(evaluator);
        server.start();
    }
}
