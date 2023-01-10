package com.ibm.oa;

import java.net.URI;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.jena.sparql.function.library.print;

import de.uni_mannheim.informatik.dws.melt.matching_base.external.http.MatcherHTTPCall;
import de.uni_mannheim.informatik.dws.melt.matching_data.SealsTrack;
import de.uni_mannheim.informatik.dws.melt.matching_data.TestCase;
import de.uni_mannheim.informatik.dws.melt.matching_data.TrackRepository;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResultSet;
import de.uni_mannheim.informatik.dws.melt.matching_eval.Executor;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorCSV;

public class MeltWebApiEvaluator {

    MeltWebApiEvaluator(String apiUri) throws Exception {
        this.trackNameToTrackMap = new HashMap<>();
        this.trackNameToTrackMap.put("conference", (SealsTrack) TrackRepository.Conference.V1);
        this.trackNameToTrackMap.put("conference_all", (SealsTrack) TrackRepository.Conference.V1_ALL_TESTCASES);
        this.trackNameToTrackMap.put("anatomy", (SealsTrack) TrackRepository.Anatomy.Default);

        this.matcher = new MatcherHTTPCall(new URI(apiUri), true);
    }

    MeltWebApiEvaluator() throws Exception {
        this("http://127.0.0.1:8081/inference/match");
    }

    private Map<String, SealsTrack> trackNameToTrackMap;
    private MatcherHTTPCall matcher;

    public Set<String> getAvailableTracks() {
        return this.trackNameToTrackMap.keySet();
    }

    public void runEvaluation(String trackName) {
        runEvaluation(trackName, null);
    }

    public void runEvaluation(String trackName, String resultsBaseDir) {
        SealsTrack track = this.trackNameToTrackMap.get(trackName);

        System.out.println("Running MELT Evaluation on track: " + trackName);
        System.out.println("Name and Version: " + track.getNameAndVersionString());
        System.out.println("Remote Location: " + track.getRemoteLocation());
        System.out.println("Number of TestCases: " + track.getTestCases().size());

        ExecutionResultSet ers = Executor.run(track, matcher);

        EvaluatorCSV evaluatorCSV = new EvaluatorCSV(ers);
        if (resultsBaseDir == null)
            evaluatorCSV.writeToDirectory();
        else
            evaluatorCSV.writeToDirectory(resultsBaseDir);
            System.out.println("Writing MELT Evaluation to: " + resultsBaseDir);
    }

    public static void main(String[] args) throws Exception {
        MeltWebApiEvaluator evaluator = new MeltWebApiEvaluator();
        evaluator.runEvaluation("anatomy");
    }

}
