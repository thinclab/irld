module runSession_UnknownObsModRobustIRL;

import sortingMDP;
import mdp;
import std.stdio;
import std.random;
import std.math;
import std.range;
import std.traits;
import std.numeric;
import std.format;
import std.algorithm;
import std.string;
import core.stdc.stdlib : exit;
import std.file;
import std.conv;
import solverApproximatingObsModel;
import irl;
import std.datetime; 

int main() {

	LinearReward reward;
	Model model;
	double chanceNoise = 0.0;
    model = new sortingMDPWdObsFeatures(0.05,null, 0, chanceNoise);
    State ts = model.S()[0];
    auto features = model.obsFeatures(ts,model.A(ts)[0],ts,model.A(ts)[0]);
	debug{
		writeln("determining number of observation features ");
	}

    int numObFeatures = cast(int)(features.length);
    model.setNumObFeatures(numObFeatures);
	// reward model
	int dim = 11;
	reward = new sortingReward7(model,dim); 
    double [] reward_weights = new double[dim];
    
    //// Read Input Arguments////
	/////////////////////////////////////
    string st;
    string buf;

	bool use_frequentist_baseline;
    buf = readln();
    formattedRead(buf, "%s", &use_frequentist_baseline);
	
	// use hierarchical distribution over two values of each observation feature or 
	// a non hierarchical distribution over 1 values
	bool useHierDistr;	
    buf = readln();
    formattedRead(buf, "%s", &useHierDistr);

	// decide what values of features will help lbfgs dsitinguish events
	bool lbfgs_use_ones;
    buf = readln();
    formattedRead(buf, "%s", &lbfgs_use_ones);
	debug{
		writeln("useHierDistr ",useHierDistr);
		writeln("lbfgs_use_ones ",lbfgs_use_ones);
	}

	double [] trueDistr_obsfeatures;
	int size_Distr_obsfeatures;
	if (useHierDistr == 1) size_Distr_obsfeatures = 2*numObFeatures;
	else size_Distr_obsfeatures = numObFeatures;
	trueDistr_obsfeatures = new double[size_Distr_obsfeatures];
	trueDistr_obsfeatures[] = 0.0;
    buf = readln();
    formattedRead(buf, "[%s]", &st);
    for (int j = 0; j < size_Distr_obsfeatures-1; j++) {
        formattedRead(st,"%s, ",&trueDistr_obsfeatures[j]);
    }
    formattedRead(st,"%s",&trueDistr_obsfeatures[size_Distr_obsfeatures-1]);
    debug {
        writeln("trueDistr_obsfeatures ",trueDistr_obsfeatures);
    }

    int num_Trajsofar;
    double numSessionsSoFar;
    // previous session Weights
    double [] lastWeights;
    double [] featureExpecExpert;
    double [] featureExpecExpertfull;
    double [] foundWeightsGlbl;
    lastWeights.length = reward.dim();
    lastWeights[] = 0.0;
    featureExpecExpertfull.length = reward.dim();
    featureExpecExpertfull[] = 0.0;
    featureExpecExpert.length = reward.dim();
    featureExpecExpert[] = 0.0;
    foundWeightsGlbl.length = reward.dim();


    buf = readln();
    formattedRead(buf, "[%s]", &st);
    for (int j = 0; j < reward.dim()-1; j++) {
        formattedRead(st,"%s, ",&lastWeights[j]);
    }
    formattedRead(st,"%s",&lastWeights[reward.dim()-1]);
    debug {
        writeln("previous session Weights ",lastWeights);
    }

    buf = readln();
    formattedRead(buf, "[%s]", &st);
    for (int j = 0; j < reward.dim()-1; j++) {
        formattedRead(st,"%s, ",&featureExpecExpert[j]);
    }
    formattedRead(st,"%s",&featureExpecExpert[reward.dim()-1]);

    buf = readln();
    formattedRead(buf, "%s", &num_Trajsofar);

    buf = readln();
    formattedRead(buf, "%s", &numSessionsSoFar);    

	// learned distribution incrementally averaged over sessions 
	double [] runAvg_learnedDistr_obsfeatures, runAvg_learnedDistr_obsfeatures2;
	if (useHierDistr == 1) runAvg_learnedDistr_obsfeatures = new double[2*model.getNumObFeatures()]; 
	else runAvg_learnedDistr_obsfeatures = new double[model.getNumObFeatures()]; 
	runAvg_learnedDistr_obsfeatures[] = 0.0;
	if (useHierDistr == 1) runAvg_learnedDistr_obsfeatures2 = new double[2*model.getNumObFeatures()]; 
	else runAvg_learnedDistr_obsfeatures2 = new double[model.getNumObFeatures()]; 
	runAvg_learnedDistr_obsfeatures2[] = 0.0;

    buf = readln();
    formattedRead(buf, "[%s]", &st);
    for (int j = 0; j < numObFeatures-1; j++) {
        formattedRead(st,"%s, ",&runAvg_learnedDistr_obsfeatures[j]);
    }
    formattedRead(st,"%s",&runAvg_learnedDistr_obsfeatures[numObFeatures-1]);
    buf = readln();
    formattedRead(buf, "[%s]", &st);
    for (int j = 0; j < numObFeatures-1; j++) {
        formattedRead(st,"%s, ",&runAvg_learnedDistr_obsfeatures2[j]);
    }
    formattedRead(st,"%s",&runAvg_learnedDistr_obsfeatures2[numObFeatures-1]);

	////////////////////////////// Setting Up Parameters/////////////////////////////////////////////

	// The new sorting MDP class is specific to pip behavior and therefore 
	// have more relaxed transition dyanamics. That introduces more 
	// options in state and action combinations for introducing noise.
	double [] params_pip_sortingModelbyPSuresh4multipleInit_onlyPIP = [0.13986013986013984, 
	0.13986013986013984, 0.13986013986013984, 0.13986013986013984, 0.013986013986013986, 
	0.006993006993006993, 0.0, 0.2797202797202797, 0.0, 0.0, 0.2797202797202797];
	reward_weights[] = params_pip_sortingModelbyPSuresh4multipleInit_onlyPIP[]; 
	double[] trueWeights = reward_weights; 

    model.setReward(reward);
	reward.setParams(reward_weights);
    model.setGamma(0.99);

	// threshold for value iteration
	double VI_threshold = 0.2; 
	// time bound for value iteration
	int vi_duration_thresh_secs = 30; 
	TimedValueIteration vi = new TimedValueIteration(int.max,false,vi_duration_thresh_secs); 
	Agent policy; 
    double[State] V; 
    V = vi.solve(model, VI_threshold); 
    policy = vi.createPolicy(model, V); 

	double[State] initial;
	foreach (s; model.S()) {
		initial[s] = 1.0;
	}
	Distr!State.normalize(initial); 
	// for comparing complete obs model 
	StateAction [] all_sa_pairs;
	foreach(s;model.S()) {
		foreach(a;model.A(s)){
			all_sa_pairs ~= new StateAction(s,a);
		}
	}

	double[StateAction][StateAction] trueObsMod;
	trueObsMod = createObsModel(model, trueDistr_obsfeatures, lbfgs_use_ones, useHierDistr); 
    
	double [] learnedDistr_obsfeatures;
	if (useHierDistr == 1) learnedDistr_obsfeatures = new double[2*model.getNumObFeatures()]; 
	else learnedDistr_obsfeatures = new double[model.getNumObFeatures()]; 

	double [] cum_prob_fvs; 
	int [][] possible_obs_fvs; 
	int [] obs_fv; 
	double prod; 
	foreach(s;model.S()) {
		foreach(a;model.A(s)){
			foreach(os;model.S()) {
				foreach(oa;model.A(os)){

					obs_fv = model.obsFeatures(s, a, os, oa);
					if (! possible_obs_fvs.canFind(obs_fv)) {
						possible_obs_fvs ~= obs_fv;
						prod = 1.0;
						foreach(i,f;obs_fv) { 
							if (lbfgs_use_ones == 1) { 
								if (f==1) prod *= trueDistr_obsfeatures[i];
							} else { 
								if (f==0) { 
									if (useHierDistr == 1) prod *= trueDistr_obsfeatures[i+model.getNumObFeatures()];
									else prod *= 1-trueDistr_obsfeatures[i];
								}
							}
						}
						cum_prob_fvs ~= prod;
					}

				}
			} 
		}
	}

	sar [][] GT_trajs;
	sac [][] obs_trajs;
	sar [] temp_traj;
	sac [] obs_traj;
	// number of trajectories input in one session 
	int num_trajs = 1;
	int size_traj = 2;
	int num_trials_perSession = 1;
	// baseline when input is noise free
	bool allGTdata = 0;
	// max length of each input trajectory
	size_t max_sample_length = 50;
	// threshold for convergence of each session, stdev of moving window of 
	// (E[phi]-hat-phi) from outputs in consecutive sessions
	double conv_threshold_stddev_diff_moving_wdw;
    // gibbs sampling
    conv_threshold_stddev_diff_moving_wdw = 0.05; 
	// number of random restarts in each session
	int restart_attempts = 1;
	// length of moving window (E[phi]-hat-phi) from outputs in consecutive sessions
	int moving_window_length_muE = 3;
	// threshold for convergence in sampling 
	double conv_threshold_gibbs = 0.015;
	// threshold for convergence in descent 
	double grad_descent_threshold = 0.0000001; // Not being used in current descent method 
	double gradient_descent_step_size = 0.00001;
	// time bound on descent
	int descent_duration_thresh_secs = 3*60;
	// threshold for approximating Ephi before descent
	double Ephi_thresh = 0.1;
	// number of samples of trajectories needed to approximate the traj-space for computing Ephi
	int nSamplesTrajSpace = 2000;
	int max_iter_lbfgs = 10000; 
	double error_lbfgs = .00001;
	int linesearch_algo = 0;
    int use_ImpSampling = 0;

	double LBA;
	double [] arr_LBA, arr_EVD; 
	arr_LBA.length = 0;
	arr_EVD.length = 0;

	double diff_wrt_muE_wo_sc;
	double diff_wrt_muE_scores1;
	double last_val;
	string base_dir = "/home/katy/Desktop/Results_RI2RL/";

	MaxEntUnknownObsModRobustIRL robustIRLUknowObsMod = new MaxEntUnknownObsModRobustIRL(restart_attempts, 
		new TimedValueIteration(int.max, false, vi_duration_thresh_secs), model.S(), 
		nSamplesTrajSpace, grad_descent_threshold, VI_threshold, 
		0.01, 100, max_iter_lbfgs, error_lbfgs, linesearch_algo);

    ////////////////////////////////// Sample Demonstration  /////////////////////////////
	int simulated_data_or_stored_data = 1;

	if (simulated_data_or_stored_data==0) {

		for(int i = 0; i < num_trajs; i++) {

			if (allGTdata==0) {
				temp_traj = simulate(model, policy, initial, size_traj);
				GT_trajs ~= temp_traj;
			} else {
				temp_traj.length = 0;
			}

			obs_traj.length = 0; 

			foreach(e_sar;temp_traj) {

				// give specific prob value to each obs feature 
				obs_fv = model.obsFeatures(e_sar.s,e_sar.a,e_sar.s,e_sar.a);

				//// introduce meaningless noise: ////
				StateAction newsa = model.noiseIntroduction(e_sar.s,e_sar.a);
				if ((newsa.s!=e_sar.s) || (newsa.a!=e_sar.a)) {
					obs_fv = model.obsFeatures(e_sar.s,e_sar.a,newsa.s,newsa.a);
				}

				obs_traj ~= sac(newsa.s,newsa.a,cum_prob_fvs[countUntil(possible_obs_fvs,obs_fv)]);
			}
			
			if (allGTdata==1) {
				foreach(s;model.S()){
					foreach(a;model.A(s)){
						temp_traj ~= sar(s,a,1.0);
						obs_traj ~= sac(s,a,1.0); 
					}
				}
				GT_trajs ~= temp_traj;
			} 

			obs_trajs ~= obs_traj;
		}
	} else {
		
		// GT_trajs must not be used when stored dataset is used for running experiments
		temp_traj = simulate(model, policy, initial, size_traj);
		GT_trajs ~= temp_traj;

		// recording observations and prediction score in file
		obs_traj.length = 0; 
		File filedata = File("/home/psuresh/Downloads/Dataset.txt", "r"); 
		// read and convert from size_traj*(numSessionsSoFar) until size_traj*(numSessionsSoFar+1) 
		int curr_ind = 0;
		int ol;
		int pr;
		int el;
		int ls;
		string action;
		Action a;
		double c;
		while (!filedata.eof()) { 
			if (curr_ind == size_traj*(numSessionsSoFar+1)) break;

			string line = chomp(filedata.readln()); 
			writeln("line -", line); 
			if (line != "ENDTRAJ") {

				if (curr_ind >= size_traj*numSessionsSoFar) {
					formattedRead(line, "[%s, %s, %s, %s];%s;%s", &ol, &pr, &el, &ls, &action, &c);
					// writeln("state components and action string - \n"); 
					// writeln(ol, pr, el, ls, action, c);

	   				if (action == "InspectAfterPicking") {
	   					a = new InspectAfterPicking();
	   				} else if (action == "InspectWithoutPicking" ) {
	   					a = new InspectWithoutPicking();
	                } else if (action == "Pick" ) {
	                    a = new Pick();
	                } else if (action == "PlaceOnConveyor" ) {
	                    a = new PlaceOnConveyor();
	                } else if (action == "PlaceInBin" ) {
	                    a = new PlaceInBin();
	                } else if (action == "ClaimNewOnion" ) {
	                    a = new ClaimNewOnion();
	   				} else if (action == "PlaceInBinClaimNextInList") {
	   					a = new PlaceInBinClaimNextInList();
	                } else {
	                    a = new ClaimNextInList();
	   				}

					obs_traj ~= sac(new sortingState([ol,pr,el,ls]),a,c);
					// writeln("obs_traj",obs_traj);
				}
				curr_ind += 1;				
			}
		}
		filedata.close(); 

		obs_trajs ~= obs_traj;
		writeln("obs_trajs ",obs_trajs);

		// writeln("read the file");
		// exit(0);
	}

    debug{
        writeln("created observed  trajectories ");
        //exit(0);
    }
    
    ///////////////////////////// Approximate Obs Model using MaxEnt ///////////////////

    double avg_cum_diff1, avg_cum_diff2, avg_cum_diff3;
    writeln("numSessionsSoFar ",numSessionsSoFar);
    learnedDistr_obsfeatures = 
    avg_singleSession_obsModelLearning(robustIRLUknowObsMod, num_trials_perSession, model,
    trueDistr_obsfeatures, GT_trajs,
    obs_trajs, trueObsMod, numSessionsSoFar,  runAvg_learnedDistr_obsfeatures,
    avg_cum_diff1, avg_cum_diff2, lbfgs_use_ones, 
    all_sa_pairs, avg_cum_diff3, useHierDistr, use_frequentist_baseline, 
	runAvg_learnedDistr_obsfeatures2, base_dir);

    // updating global variable for runing average
    // runAvg_learnedDistr_obsfeatures[] = (runAvg_learnedDistr_obsfeatures[]*(numSessionsSoFar-1) + learnedDistr_obsfeatures[]);
    // runAvg_learnedDistr_obsfeatures[] /= numSessionsSoFar; 
	runAvg_learnedDistr_obsfeatures[] = learnedDistr_obsfeatures[]; 
    double[StateAction][StateAction] obsModel = createObsModel(model, runAvg_learnedDistr_obsfeatures, lbfgs_use_ones, useHierDistr); 
    model.setObsMod(obsModel);	

    //////////////////// IRL under noisy Obs /////////////////////////////////////


    policy = robustIRLUknowObsMod.solve(model, initial, obs_trajs, max_sample_length, 
        lastWeights, last_val, foundWeightsGlbl, featureExpecExpert, num_Trajsofar, 
        Ephi_thresh, gradient_descent_step_size, descent_duration_thresh_secs,
        trueWeights, conv_threshold_stddev_diff_moving_wdw, 
        moving_window_length_muE, use_ImpSampling, conv_threshold_gibbs,
        diff_wrt_muE_wo_sc, diff_wrt_muE_scores1); 

    reward.setParams(trueWeights); 
    Agent truePolicy = vi.createPolicy(model,vi.solve(model, 0.1)); 
    reward.setParams(foundWeightsGlbl);	
    Agent learnedPolicy = vi.createPolicy(model,vi.solve(model, 0.1)); 
    LBA = robustIRLUknowObsMod.computeLBA(cast(MapAgent)learnedPolicy,cast(MapAgent)truePolicy); 

    writeln("\n LBA",LBA,"ENDLBA"); 
    writeln("\n DIFF1",diff_wrt_muE_wo_sc,"ENDDIFF1"); 
    writeln("\n DIFF2",diff_wrt_muE_scores1,"ENDDIFF2"); 


	/////////////////////////////////////////////// 
	writeln("\n\n\n\n writing results to noisyObsRobustSamplingMeirl_LBA_data \n\n\n\n ");
	
	File file1 = File("/home/psuresh/Downloads/noisyObsRobustSamplingMeirl_LBA_data_persession.csv", "a"); 
	string str_LBA = to!string(LBA);
	file1.writeln(str_LBA);
	file1.close(); 

	writeln("BEGPARSING");	

	writeln("POLICY");
	foreach (State s; model.S()) {
		foreach (Action a, double chance; policy.actions(s)) {

            sortingState ps = cast(sortingState)s;
            writeln( ps.toString(), " = ", a);
            string str_s="";
			if (ps._onion_location == 0) str_s=str_s~"Onconveyor,";
			if (ps._onion_location == 1) str_s=str_s~"Infront,";
			if (ps._onion_location == 2) str_s=str_s~"Inbin,";
			if (ps._onion_location == 3) str_s=str_s~"Picked/AtHomePose,";
			if (ps._onion_location == 4) str_s=str_s~"Placed,";

			if (ps._prediction == 0) str_s=str_s~"bad,";
			if (ps._prediction == 1) str_s=str_s~"good,";
			if (ps._prediction == 2) str_s=str_s~"unknown,";

			if (ps._EE_location == 0) str_s=str_s~"Onconveyor,";
			if (ps._EE_location == 1) str_s=str_s~"Infront,";
			if (ps._EE_location == 2) str_s=str_s~"Inbin,";
			if (ps._EE_location == 3) str_s=str_s~"Picked/AtHomePose,";

			if (ps._listIDs_status == 0) str_s=str_s~"Empty";
			if (ps._listIDs_status == 1) str_s=str_s~"NotEmpty"; 
			if (ps._listIDs_status == 2) str_s=str_s~"Unavailable";

            // writeln(str_s," = ", a);

		}
	}
	
	writeln("ENDPOLICY");

    writeln("\n WEIGHTS",foundWeightsGlbl,"ENDWEIGHTS"); 
    writeln("\n FE",featureExpecExpert,"ENDFE"); 
    writeln("\n NUMTR",num_Trajsofar,"ENDNUMTR"); 
    writeln("\n NUMSS",numSessionsSoFar,"ENDNUMSS"); 
    writeln("\n RUNAVGPTAU",runAvg_learnedDistr_obsfeatures,"ENDRUNAVGPTAU\n"); 
    writeln("\n RUNAVGPTAU2",runAvg_learnedDistr_obsfeatures2,"ENDRUNAVGPTAU2\n"); 

	debug {
		//writeln("\nSimulation for learned policy:");
		sar [] traj;
		for(int i = 0; i < 2; i++) {
			traj = simulate(model, policy, initial, 50);
			foreach (sar pair ; traj) {
				writeln(pair.s, " ", pair.a, " ", pair.r);
			}
			writeln(" ");
		}
	} 	

	writeln("ENDPARSING");

	delete reward_weights;
	delete params_pip_sortingModelbyPSuresh4multipleInit_onlyPIP;
	delete trueWeights;
	foreach (key, value; V) V.remove(key);
	foreach (key, value; initial) initial.remove(key);
	delete GT_trajs;
	delete obs_trajs;
	delete temp_traj;
	delete obs_traj;
	delete trueDistr_obsfeatures;
	delete cum_prob_fvs; 
	delete possible_obs_fvs; 
	delete obs_fv; 
	foreach (key, value; trueObsMod) trueObsMod.remove(key);
	delete learnedDistr_obsfeatures;
	delete runAvg_learnedDistr_obsfeatures;
	delete runAvg_learnedDistr_obsfeatures2;
	delete arr_LBA;
	delete arr_EVD; 
	delete lastWeights;
	delete featureExpecExpert;
	delete featureExpecExpertfull;
	delete foundWeightsGlbl;

    return 0;

}