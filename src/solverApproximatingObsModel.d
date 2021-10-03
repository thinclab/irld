import std.stdio;
import mdp;
import irl;
import std.random;
import std.math;
import std.range;
import std.traits;
import std.numeric;
import std.format;
import std.algorithm;
import std.string;
import core.stdc.stdlib : exit;
import std.string;
import std.conv;

public double [] simulateNoisyDemo_Incremental_ObsModLearning(Model model, StateAction [] all_sa_pairs, Agent policy, 
	double[State] initial, int num_trajs, int size_traj, bool useHierDistr, bool lbfgs_use_ones, 
	int num_sessions, int num_trials_perSession, bool allGTdata, ref double[][] arr_metric1data, 
	ref double[][] arr_metric3data, bool use_frequentist_baseline, string base_dir) {

	sar [][] GT_trajs;
	sac [][] obs_trajs;
	sar [] temp_traj;
	sac [] obs_traj;
	// collect all possible values of arrays output of features function

	// Option 1 for testing efficacy of approximation: Create random distr over features. assume multiplicative model for computing 
	// the cumulative probs for all possible combinations of feature values
	double [] trueDistr_obsfeatures;
	if (useHierDistr == 1) trueDistr_obsfeatures = new double[2*model.getNumObFeatures()];
	else trueDistr_obsfeatures = new double[model.getNumObFeatures()];

	double currp;
	double totalmass_untilNw = 0.0;
	foreach (i; 0 .. model.getNumObFeatures()) {
		if (useHierDistr == 1) currp = uniform(0.0,1);
		else {
			currp = uniform(0.0,1-totalmass_untilNw);
			totalmass_untilNw += currp;
		}
		trueDistr_obsfeatures[i] = currp;
		if (useHierDistr == 1) trueDistr_obsfeatures[i+model.getNumObFeatures()] = 1-currp;
	} 
	if (useHierDistr == 0) {
		trueDistr_obsfeatures[model.getNumObFeatures()-1] += 1-totalmass_untilNw;
	}

	debug { 
		writeln("true distr obsfeatures ",trueDistr_obsfeatures); 
		//exit(0);
	}

	// Option 2: HARDCODE the cumulative probs for all possible combinations of feature values
	//cum_prob_fvs = [0.1,0.3,0.6];

	double [] cum_prob_fvs;
	int [][] possible_obs_fvs;
	int [] obs_fv;
	double prod;
	//int idx_fv = 0;

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

	double[StateAction][StateAction] trueObsMod;
	trueObsMod = createObsModel(model, trueDistr_obsfeatures, lbfgs_use_ones, useHierDistr); 

	double [] learnedDistr_obsfeatures;
	if (useHierDistr == 1) learnedDistr_obsfeatures = new double[2*model.getNumObFeatures()]; 
	else learnedDistr_obsfeatures = new double[model.getNumObFeatures()]; 

	// learned distribution incrementally averaged over sessions 
	double [] runAvg_learnedDistr_obsfeatures, runAvg_learnedDistr_obsfeatures2;
	if (useHierDistr == 1) runAvg_learnedDistr_obsfeatures = new double[2*model.getNumObFeatures()]; 
	else runAvg_learnedDistr_obsfeatures = new double[model.getNumObFeatures()]; 
	runAvg_learnedDistr_obsfeatures[] = 0.0;
	if (useHierDistr == 1) runAvg_learnedDistr_obsfeatures2 = new double[2*model.getNumObFeatures()]; 
	else runAvg_learnedDistr_obsfeatures2 = new double[model.getNumObFeatures()]; 
	runAvg_learnedDistr_obsfeatures2[] = 0.0;

	double numSessionsSoFar = 0.0;
	// arrays of error values for all metrics 
	double [] arr_avg_cum_diff1, arr_avg_cum_diff2, arr_avg_cum_diff3;

	//MaxEntUnknownObsMod estimateObsMod = new MaxEntUnknownObsMod(100,new ValueIteration(), 2000, .00001, .1, .1);

	// number of random restarts in each session
	int restart_attempts = 1;
	// number of samples of trajectories needed to approximate the traj-space for computing Ephi
	int nSamplesTrajSpace = 2000;
	// threshold for convergence in descent 
	double grad_descent_threshold = 0.0000001; // Not being used in current descent method 
	// threshold for value iteration
	double VI_threshold = 0.1; 
	// time bound for value iteration
	int vi_duration_thresh_secs = 30; 
	// int max_iter_lbfgs = 1000000; 
	// int max_iter_lbfgs = 1; 
	// int max_iter_lbfgs = 2; 
	int max_iter_lbfgs = 5; 
	double error_lbfgs = 0.00001;
	int linesearch_algo = 0;
	// int linesearch_algo = 1;
	// int linesearch_algo = 2;
	// int linesearch_algo = 3;

	/*
		this(int max_iter, MDPSolver solver, State [] observableStates, int n_samples=500, 
		double error=0.1, double solverError =0.1, double Qsolve_qval_thresh = 0.01, 
		ulong QSolve_max_iter = 100, int max_iter_lbfgs=100, double error_lbfgs=.00001, linesearch_algo) {
		super(max_iter, solver, observableStates, n_samples, error, solverError,
			Qsolve_qval_thresh, QSolve_max_iter);	
	*/
	MaxEntUnknownObsModRobustIRL estimateObsMod = new MaxEntUnknownObsModRobustIRL(restart_attempts, 
		new TimedValueIteration(int.max, false, vi_duration_thresh_secs), 
		model.S(), nSamplesTrajSpace, grad_descent_threshold, VI_threshold, 
		0.01, 100, max_iter_lbfgs, error_lbfgs, linesearch_algo);

	foreach(idxs; 0..num_sessions) {
		////////////////////////////////// Demonstration  /////////////////////////////


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

				//// add meaningless noise: ////
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

		debug{
			writeln("created observed  trajectories ");
			//exit(0);
		}


		////////////////////////////////// Session Starts /////////////////////////////
		double avg_cum_diff1, avg_cum_diff2, avg_cum_diff3;
		
		writeln("numSessionsSoFar ",numSessionsSoFar);
		learnedDistr_obsfeatures = 
		avg_singleSession_obsModelLearning(estimateObsMod, num_trials_perSession, model,
		trueDistr_obsfeatures, GT_trajs,
		obs_trajs, trueObsMod, numSessionsSoFar,  runAvg_learnedDistr_obsfeatures,
		avg_cum_diff1, avg_cum_diff2, lbfgs_use_ones, 
		all_sa_pairs, avg_cum_diff3, useHierDistr,
		use_frequentist_baseline,  runAvg_learnedDistr_obsfeatures2, base_dir);


		arr_avg_cum_diff1 ~= avg_cum_diff1; 
		arr_avg_cum_diff2 ~= avg_cum_diff2; 
		writeln("arr_avg_cum_diff1 ",arr_avg_cum_diff1);
		arr_avg_cum_diff3 ~= avg_cum_diff3;

		///////////////////////////////////////// Session Finished /////////////////////////////////////////////

		runAvg_learnedDistr_obsfeatures[] = learnedDistr_obsfeatures[];
		// updating global variable for runing average
		// runAvg_learnedDistr_obsfeatures[] = (runAvg_learnedDistr_obsfeatures[]*(numSessionsSoFar-1) + learnedDistr_obsfeatures[]);
		// runAvg_learnedDistr_obsfeatures[] /= numSessionsSoFar; 

	}

	//writeln(arr_avg_cum_diff1, arr_avg_cum_diff2, arr_avg_cum_diff3); 
	//writeln(runAvg_learnedDistr_obsfeatures); 
	debug{
		// writeln("writing ",arr_avg_cum_diff1," to resultsApproxObsModelMetric1 ");
	}
	File file1 = File(base_dir~"resultsApproxObsModelMetric1.csv", "a"); 
	string str_arr_avg_cum_diff1 = to!string(arr_avg_cum_diff1);
	str_arr_avg_cum_diff1 = str_arr_avg_cum_diff1[1 .. (str_arr_avg_cum_diff1.length-1)];
	file1.writeln(str_arr_avg_cum_diff1);
	file1.close(); 
	arr_metric1data ~= arr_avg_cum_diff1;

	File file2 = File(base_dir~"resultsApproxObsModelMetric2.csv", "a"); 
	string str_arr_avg_cum_diff2 = to!string(arr_avg_cum_diff2);
	str_arr_avg_cum_diff2 = str_arr_avg_cum_diff2[1 .. (str_arr_avg_cum_diff2.length-1)];
	file2.writeln(str_arr_avg_cum_diff2);
	file2.close(); 

	File file3 = File(base_dir~"resultsApproxObsModelMetric3.csv", "a"); 
	string str_arr_avg_cum_diff3 = to!string(arr_avg_cum_diff3);
	str_arr_avg_cum_diff3 = str_arr_avg_cum_diff3[1 .. (str_arr_avg_cum_diff3.length-1)];
	file3.writeln(str_arr_avg_cum_diff3);
	file3.close(); 
	arr_metric3data ~= arr_avg_cum_diff3;

	return runAvg_learnedDistr_obsfeatures;
}


public double[StateAction][StateAction] createObsModel(Model model, double [] featureWeights, 
	bool lbfgs_use_ones, bool useHierDistr) {

	double p_success, tot_mass_obsSA_wd_sharedActvFeat, temp_total_mass;

	int [] features;
	double[StateAction][StateAction] returnedObModel;
	double[StateAction] tempdict_gt_sa;
	int totalSA_pairs, tot_obsSA_wo_sharedActvFeat;

	totalSA_pairs = 0;
	foreach(s; model.S()) { // for each state
		foreach(a; model.A(s)){
			totalSA_pairs += 1;
		}
	}
	int num_keys_obsMod = 0;
	int num_keys_obsMod_currgtsa = 0;

	foreach(s; model.S()) { // for each state
		foreach(a; model.A(s)){
			StateAction curr_gt_sa = new StateAction(s,a);
			debug {
				//writeln("createObsModel: curr_gt_sa ",curr_gt_sa);
			} 

			num_keys_obsMod_currgtsa = 0;
			tot_mass_obsSA_wd_sharedActvFeat = 0.0;
			tot_obsSA_wo_sharedActvFeat = 0;

			foreach(obs_s; model.S()) { // for each obs state
				foreach(obs_a; model.A(obs_s)){ 
					StateAction obs_gt_sa = new StateAction(obs_s,obs_a);
					debug {
						//writeln("createObsModel: obs_gt_sa ",obs_gt_sa);
					} 
					// use p_success of the features shared among GT and obs  

					p_success = 1.0;
					features = model.obsFeatures(s,a,obs_s,obs_a);

					// P(obs s, obs a | s, a) = 
					foreach(i,f;features) {
						if (lbfgs_use_ones == 1) {
							if (f == 1) p_success *= featureWeights[i];
						} else {
							if (f==0) { 
								if (useHierDistr == 1) p_success *= featureWeights[i+model.getNumObFeatures()];
								else p_success *= 1-featureWeights[i];
							}
						}
					} 

					if (p_success < 0) p_success = 0;
					if (p_success > 1) p_success = 1;
					
					// if no features activated (p_success == 1.0) 
					// or features could not capture the physical description in that s-a 
					// Option 1 uniform distribution with known denominator
					if (p_success == 1.0) p_success = 1.0/cast(double)totalSA_pairs;
					// Option 1 uniform distribution with denominator yet to be known
					//if (p_success == 1.0) tot_obsSA_wo_sharedActvFeat += 1;
					//else  tot_mass_obsSA_wd_sharedActvFeat += p_success;

					returnedObModel[curr_gt_sa][obs_gt_sa] = p_success;
					num_keys_obsMod_currgtsa += 1;
				}
			}
			
			num_keys_obsMod += 1;

			Distr!StateAction.normalize(returnedObModel[curr_gt_sa]);

			debug {
				// writeln (returnedObModel[new StateAction(s,a)]); 
				// test if you can sample from the distribution 
				tempdict_gt_sa = returnedObModel[curr_gt_sa];
				StateAction sampled_obs_sa = Distr!StateAction.sample(tempdict_gt_sa);
				// writeln("test sampling observation ",sampled_obs_sa);
				bool allKeyIn = (num_keys_obsMod_currgtsa==totalSA_pairs);
				if (!allKeyIn) writeln("num_keys_obsMod_currgtsa == totalSA_pairs ",(num_keys_obsMod_currgtsa==totalSA_pairs));
			}
		}
	}

	debug {

		bool allKeyIn = (num_keys_obsMod==totalSA_pairs);
		if (!allKeyIn) writeln("num_keys_obsMod == totalSA_pairs ",(num_keys_obsMod==totalSA_pairs));
		
	}
	return returnedObModel;

} 

public double [] avg_singleSession_obsModelLearning(MaxEntUnknownObsModRobustIRL estimateObsMod,
	int num_trials_perSession, Model model, 
	double [] trueDistr_obsfeatures, sar [][] GT_trajs,
	sac [][] obs_trajs, double[StateAction][StateAction] trueObsMod,
	ref double numSessionsSoFar, ref double [] runAvg_learnedDistr_obsfeatures, 
	ref double avg_cum_diff1, ref double avg_cum_diff2, 
	bool lbfgs_use_ones, StateAction[] all_sa_pairs, ref double avg_cum_diff3,
	bool useHierDistr, bool use_frequentist_baseline, ref double [] runAvg_learnedDistr_obsfeatures2,
	string base_dir) {
	//////// For average over 10 runs with same trueObsFeatDistr and observations /////// 
	
	writeln("numSessionsSoFar ",numSessionsSoFar);
	double [] arr_cum_diff1, arr_cum_diff2, arr_cum_diff3;
	// array of outputs from trials within same session
	double [] learnedDistr_obsfeatures;
	double [][] arr_learnedDistr_obsfeatures;
	arr_learnedDistr_obsfeatures.length = 0;
	arr_cum_diff1.length = 0;
	arr_cum_diff2.length = 0;
	arr_cum_diff3.length = 0;
	numSessionsSoFar += 1; 
	double opt_val_Obj;

	foreach(tr; 0..num_trials_perSession) {

		if (use_frequentist_baseline == true) {
			learnedDistr_obsfeatures = estimateObsMod.frequentistEstimateDistrObsModelFeatures(model, obs_trajs, useHierDistr);
			// learnedDistr_obsfeatures2 = estimateObsMod.learnDistrObsModelFeatures(model, obs_trajs, opt_val_Obj, lbfgs_use_ones, useHierDistr); 
		} else {
			learnedDistr_obsfeatures = estimateObsMod.learnDistrObsModelFeatures(model, obs_trajs, opt_val_Obj, lbfgs_use_ones, useHierDistr);
			// learnedDistr_obsfeatures2 = estimateObsMod.frequentistEstimateDistrObsModelFeatures(model, obs_trajs, useHierDistr);
		}

		// update the incrementally learned feature distribution 
		// local substitute variable for incremental runing average
		double [] temp_runAvg_learnedDistr_obsfeatures = new double[learnedDistr_obsfeatures.length];
		//writeln("runAvg_learnedDistr_obsfeatures ",runAvg_learnedDistr_obsfeatures);
		//writeln("numSessionsSoFar ",numSessionsSoFar);

		// Updating the running average 
		temp_runAvg_learnedDistr_obsfeatures[] = (runAvg_learnedDistr_obsfeatures[]*(numSessionsSoFar-1) + learnedDistr_obsfeatures[]);
		temp_runAvg_learnedDistr_obsfeatures[] /= numSessionsSoFar; 

		// normalization should happen after running average 
		if (useHierDistr==1) {
			foreach (i; 0..(temp_runAvg_learnedDistr_obsfeatures.length/2)) {
				debug {
					// writeln("\n Q before normalizing ",temp_runAvg_learnedDistr_obsfeatures[i]," ",temp_runAvg_learnedDistr_obsfeatures[i+model.getNumObFeatures()]);
				}
				// separately for each tau and taubar
				double p_psi = temp_runAvg_learnedDistr_obsfeatures[i]/(temp_runAvg_learnedDistr_obsfeatures[i]
					+temp_runAvg_learnedDistr_obsfeatures[i+model.getNumObFeatures()]);
				double p_psi_bar = temp_runAvg_learnedDistr_obsfeatures[i+model.getNumObFeatures()]/(temp_runAvg_learnedDistr_obsfeatures[i]
					+temp_runAvg_learnedDistr_obsfeatures[i+model.getNumObFeatures()]);
				temp_runAvg_learnedDistr_obsfeatures[i] = p_psi;
				temp_runAvg_learnedDistr_obsfeatures[i+model.getNumObFeatures()] = p_psi_bar;
				debug {
					// writeln("\n Q after normalizing ",temp_runAvg_learnedDistr_obsfeatures[i]," ",temp_runAvg_learnedDistr_obsfeatures[i+model.getNumObFeatures()]);
				}
			}
		} else {
			temp_runAvg_learnedDistr_obsfeatures[] = temp_runAvg_learnedDistr_obsfeatures[]/sum(temp_runAvg_learnedDistr_obsfeatures);
		}

		// for averaging over trials within session
		arr_learnedDistr_obsfeatures ~= temp_runAvg_learnedDistr_obsfeatures;

		double[StateAction][StateAction] learnedObsMod;
		learnedObsMod = createObsModel(model, temp_runAvg_learnedDistr_obsfeatures, lbfgs_use_ones, useHierDistr); 

		double diffprod, cum_diff1, cum_diff2, cum_diff1a, prod;
		// Metric 1a:
		// Cumulative total diff in true and learned distribution
		double diff1;
		double [] tempdiff1, tempdiff2; 
		cum_diff1 = 0.0;
		foreach (i; 0 .. model.getNumObFeatures()) {

			if (useHierDistr == 1) {
				// Using KL Divergence
				double [] P = new double[2];
				P[0] = trueDistr_obsfeatures[i];
				P[1] = trueDistr_obsfeatures[i+model.getNumObFeatures()];
				double [] Q = new double[2];
				Q[0] = temp_runAvg_learnedDistr_obsfeatures[i];
				Q[1] = temp_runAvg_learnedDistr_obsfeatures[i+model.getNumObFeatures()];
				diff1 = KL_divergence(P,Q);
				debug {
					// writeln("\n P ",P,"\n Q ",Q,"\n KLD ",diff1);
				}

				// Using Euclidean Distance
				// tempdiff1 ~= trueDistr_obsfeatures[i]; 
				// tempdiff1 ~= trueDistr_obsfeatures[i+model.getNumObFeatures()]; 
				// tempdiff2 ~= temp_runAvg_learnedDistr_obsfeatures[i]; 
				// tempdiff2 ~= temp_runAvg_learnedDistr_obsfeatures[i+model.getNumObFeatures()]; 
				// tempdiff1[] = tempdiff1[]-tempdiff2[];
				// diff1 = l1norm(tempdiff1)/2; 
				// tempdiff1.length = 0; 
				// tempdiff2.length = 0; 

			} else {
				if (lbfgs_use_ones) diff1 = trueDistr_obsfeatures[i]-temp_runAvg_learnedDistr_obsfeatures[i]; 
				else diff1 = trueDistr_obsfeatures[i+model.getNumObFeatures()]
					-temp_runAvg_learnedDistr_obsfeatures[i+model.getNumObFeatures()]; 
			}

			cum_diff1 += diff1;
			//writeln(diff1,"  ",cum_diff1);

		} 
		cum_diff1 = cum_diff1/cast(double)model.getNumObFeatures();
		arr_cum_diff1 ~= cum_diff1;
		//exit(0);

		// Metric 1b: 
		// Cumulative diff for P(obs-sa | GT-sa) distributions for only those GT-sa pairs that got corrupted in input to perception pipeline
		//StateAction[] corrupted_GT_SAs;
		//corrupted_GT_SAs.length = 0;
		//bool corrupted;
		//foreach(i,GT_traj;GT_trajs) {
		//	sac[] ob_traj = obs_trajs[i];
		//	foreach(j,e_sar;GT_traj) {
		//		sac e_sac = ob_traj[j];
		//		if ((e_sar.s != e_sac.s) || (e_sar.a != e_sac.a)) {
		//			StateAction corrupted_sa = new StateAction(e_sar.s,e_sar.a);
		//			// not demonstrated
		//			if (! corrupted_GT_SAs.canFind(corrupted_sa)) corrupted_GT_SAs ~= corrupted_sa;
		//			//writeln("corrupted_GT_SAs ",corrupted_GT_SAs);
		//		}
		//	}
		//}
		//cum_diff1 = 0.0;
		//foreach(sa;corrupted_GT_SAs){
		//	cum_diff1 += normedDiff_SA_Distr(trueObsMod[sa], learnedObsMod[sa]);
		//}
		//if (cast(double)corrupted_GT_SAs.length == 0) {
		//	cum_diff1 = -double.max;
		//} else {
		//	cum_diff1 = cum_diff1/cast(double)corrupted_GT_SAs.length;
		//}
		//arr_cum_diff1 ~= cum_diff1;
		//writeln("cumulative diff for observation likelihood of s-a pairs misidentified due to noise ",cum_diff1);
		//exit(0);

		// Metric 2: estimation of the observation likelihood of s-a pairs never present in input of perception pipeline 
		// Cumulative diff for P(obs-sa | GT-sa) distributions for GT-sa pairs that never occured in input to perception pipeline
		StateAction[] unseen_GT_SAs;
		unseen_GT_SAs.length = 0;
		bool inDemonsInpNeuralNet;
		foreach(s;model.S()) {
			foreach(a;model.A(s)){
				inDemonsInpNeuralNet = false;
				foreach(GT_traj;GT_trajs) {
					foreach(e_sar;GT_traj) {

						if (e_sar.s.opEquals(s) && e_sar.a.opEquals(a)) {
							inDemonsInpNeuralNet = true;
						} 
					}
				}

				if (inDemonsInpNeuralNet == false) {
					StateAction curr_gt_sa = new StateAction(s,a);
					//writeln("e_sar",e_sar);
					//writeln("s:",s," a:",a);
					//writeln("unseen_GT_SAs ",unseen_GT_SAs);
					// not demonstrated
					if (! unseen_GT_SAs.canFind(curr_gt_sa)) unseen_GT_SAs ~= curr_gt_sa;
				}
			}
		} 
		cum_diff2 = 0.0;
		foreach(sa;unseen_GT_SAs){
			cum_diff2 += normedDiff_SA_Distr(trueObsMod[sa], learnedObsMod[sa]);
		}
		if (cast(double)unseen_GT_SAs.length == 0) {
			cum_diff2 = -double.max;
		} else {
			cum_diff2 = cum_diff2/cast(double)unseen_GT_SAs.length;
		}
		debug writeln("Divide by 0 check: number of sa pairs:",trueObsMod.length,"\n number of unseen GT sa pairs:",unseen_GT_SAs.length);
		
		arr_cum_diff2 ~= cum_diff2;

		//Metric 3: diff for all state action pairs 
		double cum_diff3=0.0;
		foreach(sa;all_sa_pairs) {
			cum_diff3 += normedDiff_SA_Distr(trueObsMod[sa], learnedObsMod[sa]);
		} 
		cum_diff3 = cum_diff3/cast(double)all_sa_pairs.length; 
		arr_cum_diff3 ~= cum_diff3; 
	} 

	avg_cum_diff1 = (sum(arr_cum_diff1)+0.000001)/cast(double)num_trials_perSession;
	avg_cum_diff2 = (sum(arr_cum_diff2)+0.000001)/cast(double)num_trials_perSession;
	avg_cum_diff3 = (sum(arr_cum_diff3)+0.000001)/cast(double)num_trials_perSession;

	debug{

		//writeln(arr_cum_diff1,arr_cum_diff2);
		writeln("average cum_diff1 for ",num_trials_perSession," trials of this session ",avg_cum_diff1);
		writeln("average cum_diff2 for ",num_trials_perSession," trials of this session ",avg_cum_diff2);
		writeln("numSessionsSoFar ",numSessionsSoFar);
	}

	// average learned distribution from trials within session 
	foreach(i; 0 .. arr_learnedDistr_obsfeatures[0].length ) {
		learnedDistr_obsfeatures[i] = sum(arr_learnedDistr_obsfeatures.transversal(i))/cast(double)(arr_learnedDistr_obsfeatures.length);
	} 

	// Recording diff w.r.t. baseline
	double [] learnedDistr_obsfeatures2, temp_runAvg_learnedDistr_obsfeatures2;
	File fileLearnedDistrDiff = File(base_dir~"resultsLearnedDistrCurrentSession.csv", "a"); 
	if (use_frequentist_baseline == true) { 
		learnedDistr_obsfeatures2 = estimateObsMod.learnDistrObsModelFeatures(model, obs_trajs, opt_val_Obj, lbfgs_use_ones, useHierDistr); 
	} else { 
		learnedDistr_obsfeatures2 = estimateObsMod.frequentistEstimateDistrObsModelFeatures(model, obs_trajs, useHierDistr);
	} 
	temp_runAvg_learnedDistr_obsfeatures2.length = learnedDistr_obsfeatures2.length;
	temp_runAvg_learnedDistr_obsfeatures2[] = (runAvg_learnedDistr_obsfeatures2[]*(numSessionsSoFar-1) + learnedDistr_obsfeatures2[]);
	temp_runAvg_learnedDistr_obsfeatures2[] /= numSessionsSoFar; 
	runAvg_learnedDistr_obsfeatures2[] = temp_runAvg_learnedDistr_obsfeatures2[];
	double diff4, cum_diff4;
	cum_diff4 = 0.0;
	foreach (i; 0 .. model.getNumObFeatures()) {

		if (useHierDistr == 1) {
			// Using KL Divergence
			double [] P = new double[2];
			P[0] = learnedDistr_obsfeatures[i];
			P[1] = learnedDistr_obsfeatures[i+model.getNumObFeatures()];
			double [] Q = new double[2];
			Q[0] = runAvg_learnedDistr_obsfeatures2[i];
			Q[1] = runAvg_learnedDistr_obsfeatures2[i+model.getNumObFeatures()];
			diff4 = KL_divergence(P,Q);
			debug {
				// writeln("\n P ",P,"\n Q ",Q,"\n KLD ",diff1);
			}

		} else {
			if (lbfgs_use_ones) diff4 = learnedDistr_obsfeatures[i]-runAvg_learnedDistr_obsfeatures2[i]; 
			else diff4 = learnedDistr_obsfeatures[i+model.getNumObFeatures()]
				-runAvg_learnedDistr_obsfeatures2[i+model.getNumObFeatures()]; 
		}
		cum_diff4 += diff4;
	} 
	cum_diff4 = cum_diff4/cast(double)model.getNumObFeatures();

	if (numSessionsSoFar==0) fileLearnedDistrDiff.writeln("\n I2RL starts here \n");
	string stDiffDistrMethod1Method2 = to!string(cum_diff4);
	fileLearnedDistrDiff.writeln(stDiffDistrMethod1Method2);
	fileLearnedDistrDiff.close(); 

	return learnedDistr_obsfeatures;
}

double KL_divergence(double [] P, double [] Q) {
	double sum = 0.0;
	foreach (i; 0..P.length)
	{
		sum += P[i]*log(P[i]/Q[i]);
	}
	return sum;
}

string feature_vector_to_string(int [] features) {
	string returnval = "";
	foreach (f; features) {
		returnval ~= f == 0 ? "0" : "1";
	}
	return returnval;
}

int[][] define_events_obsMod(Model model, sac[][] samples, out int[string] eventMapping, 
	bool lbfgs_use_ones) {
	// vector of all the outputs of feature function, that got instantiated in samples
	// <s,a,c> c = score
	
	int[][] returnval;
	int[string] eventIds;
	

	foreach(sample; samples) {

		foreach(e_sac; sample) {

			foreach(s; model.S()) { // for each state action pair
				foreach(a; model.A(s)){

					auto features = model.obsFeatures(s,a,e_sac.s,e_sac.a);

					auto eID = feature_vector_to_string(features);

					if (! ( eID in eventIds)) {
						eventIds[eID] = cast(int)returnval.length;
						
						int[] feature_indices;
						foreach(i,f;features) {
							if (lbfgs_use_ones == 1) {
								if (f==1) feature_indices ~= cast(int)i;
							} else {
								if (f==0) feature_indices ~= cast(int)i;
							}
						} 

						// returnval [eventIds[eID]]
						returnval ~= feature_indices;

						debug {
							//writeln("define events: eID  = ",eID," features=",features);
							//writeln("returnval ",returnval);
						}
					}
				}
			}
		}
	}
	//exit(0);
	eventMapping = eventIds;
	
	return returnval;
}


double [] calc_log_success_rate_obsMod(sac[][] samples, int[string] eventMapping, Model model) {

	double [] returnval = new double[eventMapping.length];
	returnval[] = 0;
	
	int[string] totalCount;
	double[string] cumulativeScore;
	debug {
		writeln("calc_log_success_rate_obsMod: ");

	}
	
	foreach(i,sample; samples) {
		debug {
			writeln("sample ",sample );
		}
		foreach(e_sac; sample) {

			foreach(s; model.S()) { // for each state action pair
				foreach(a; model.A(s)){

					// P(obs s, obs a | s, a)

					auto features = model.obsFeatures(s,a,e_sac.s,e_sac.a);

					auto eID = feature_vector_to_string(features);

					totalCount[eID] += 1;
					cumulativeScore[eID] += e_sac.c;

				}
			}

		}
	
	}
	
	foreach (ID, num; eventMapping) {
		if (ID in cumulativeScore && ID in totalCount) {
			debug {
				//writeln("ID in cumulativeScore ",cumulativeScore[ID]," && ID in totalCount ",totalCount[ID]);
				//exit(0);
			}
			returnval[num] = log(cast(double)(cumulativeScore[ID] ) / cast(double)(totalCount[ID] ) );
		}
		else {
			returnval[num] = 0;
		}
		debug {
			//writeln(returnval[num]);
		}
	}
	//exit(0);

	return returnval;
}

int [][] coarsest_partition(int[][] E) {

	int[][] S;
	int[][] Edup = E.dup;

	// union of all members of E, i.e. all feature vectors that got instantiated in samples
	int [] trivial = set_type!(int).to_set(uniq(nWayUnion(Edup)));
	S ~= trivial;
	
	int [] empty_set;
	empty_set.length = 0;
	
	foreach (j, Ej; E) {
		foreach(i, Si; S) {
			auto intersection = setIntersection(Ej, Si);

			
			if (equal(intersection, empty_set) || equal(intersection, Si))
				continue; // Si is either a proper subset of Ej, or else has nothing in common
		
			
			S ~= set_type!(int).to_set(intersection);
			
			S[i] = set_type!(int).to_set(setDifference(Si, intersection));

		}	

	}

	
	return S;
}

template set_type(Q) {
	Q [] to_set(R)(R si) if (isInputRange!(R) && is(ElementType!(R) : Q) ) {
		
		Q [] returnval;
		foreach(s; si) {
			returnval ~= s;
		}
		return returnval;
	}
}

double [] map_p_onto_features(double [] p, int [][] S, int num_obsfeatures, 
	bool lbfgs_use_ones, bool useHierDistr) {

	double [] returnval;
	if (useHierDistr==1) returnval = new double[2*num_obsfeatures];
	else returnval = new double[num_obsfeatures];

	returnval[] = 1;
	    	
	foreach(i, si; S) {
		foreach(subsi; si) {
			if (lbfgs_use_ones==1) {
				// tau
				returnval[subsi] = pow(p[i], 1.0 / S[i].length); 
				if (useHierDistr==1) returnval[subsi+num_obsfeatures] = pow(p[i+S.length], 1.0 / S[i].length); // taubar
			} else {
				// tau
				returnval[subsi] = pow(p[i+S.length], 1.0 / S[i].length); 
				if (useHierDistr==1) returnval[subsi+num_obsfeatures] = pow(p[i], 1.0 / S[i].length); // taubar
			}
		}
	}
	
	return returnval;
}

class MaxEntUnknownObsModRobustIRL : MaxEntIrlZiebartApproxNoisyObs {

	private int max_iter_lbfgs;
	private int linesearch_algo;
	private double error_lbfgs;
	private int[][] E; // E is a set of subsets, each subset contains the number of a feature
	private double[] Q; // Q is the success rate of each E
	private int[][] S; // the `coursest partition of the feature space induced by the Es
	private int[][] P; // M x N matrix, each row contains the vector of v's corresponding to a given P
	    	
    private double [] v;
    private double [] lambda;
    private sac[][] obsTraj_samples;

	public this(int max_iter, MDPSolver solver, State [] observableStates, int n_samples=500, 
		double error=0.1, double solverError =0.1, double Qsolve_qval_thresh = 0.01, 
		ulong QSolve_max_iter = 100, int max_iter_lbfgs=100, double error_lbfgs=.00001,
		int linesearch_algo=0) {
		super(max_iter, solver, observableStates, n_samples, error, solverError,
			Qsolve_qval_thresh, QSolve_max_iter);	
		this.max_iter_lbfgs =max_iter_lbfgs;
		this.error_lbfgs = error_lbfgs;
		this.linesearch_algo = linesearch_algo;
	}

	public double [] frequentistEstimateDistrObsModelFeatures(Model model, sac[][] obsTraj_samples, bool useHierDistr) {
		// for each feature, for each observation, we assume that ground truth is same as observation and 
		// p(tau) = (number of times observation is made) / (total number of observations) 

		double [] frequencies_taus_as1;
		if (useHierDistr==1) frequencies_taus_as1.length = 2*model.getNumObFeatures();
		else frequencies_taus_as1.length = model.getNumObFeatures(); 
		frequencies_taus_as1[] = 0.0001; // to avoid divide by 0

		int [] ob_features;
		foreach (sac[] obsTraj_sample; obsTraj_samples)
		{
			foreach (sac e_sac; obsTraj_sample)
			{
				ob_features = model.obsFeatures(e_sac.s, e_sac.a, e_sac.s, e_sac.a); 
			}
			foreach (i; 0..model.getNumObFeatures())
			{
				if (ob_features[i] == 1) frequencies_taus_as1[i] += 1;
			}
		}

		double[] returnDistr = new double[frequencies_taus_as1.length];
		foreach (i; 0..model.getNumObFeatures())
		{
			returnDistr[i] = frequencies_taus_as1[i]/sum(frequencies_taus_as1);
			if (useHierDistr==1) returnDistr[i+model.getNumObFeatures()] = 1-returnDistr[i];
		}
		debug {
			// writeln("frequentistEstimateDistrObsModelFeatures ",returnDistr);
		}
		return returnDistr;
	}

	public double [] learnDistrObsModelFeatures(Model model, sac[][] obsTraj_samples, out double opt_value, 
		bool lbfgs_use_ones, bool useHierDistr) {
		
        // Compute feature expectations of agent = mu_E from samples
        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.linesearch = linesearch_algo;
        param.max_iterations = max_iter_lbfgs;
        param.epsilon = error_lbfgs;
        //param.wolfe = 0.999;
		//param.min_step = 0.00001; 
        //param.min_step = 0.1;
        
        this.model = model;
        this.obsTraj_samples = obsTraj_samples;
        this.sample_length = cast(int)obsTraj_samples.length;
        
        int[string] eventMapping;
        E = define_events_obsMod(model, obsTraj_samples, eventMapping, lbfgs_use_ones); 
        debug {
        	writeln("E: ", E);
        }
        
        Q = calc_log_success_rate_obsMod(obsTraj_samples, eventMapping, model);
        foreach (ref q; Q) {
        	q = exp(q);
        }
        
        debug {
        	writeln("Q: ", Q);
        }
        
        S = coarsest_partition(E);
        
        debug {
        	writeln("partitions S: ", S);
        }	
        
        // we need a lagrangian multiplier for each event, call it v
    	v = new double[E.length];
    	v[] = .01;
    	lambda = new double[S.length];
    	lambda[] = .0001;
    	
    	
        double * p = lbfgs_malloc(2*cast(int)S.length);
        scope(exit) {
        	lbfgs_free(p);
        } 
        
        foreach (i; 0..(2*S.length)) {
        	p[i] = uniform(0.05, .95);
	        //write(p[i]," "); 
        }

        //writeln("p ",to!string(*p));
        //exit(0);
        
        double finalValue;	
        double [] weights;	
        weights.length = 2*S.length;
        int ret;	
        foreach (i; 0..30) {

        	auto temp =this;
        	ret = lbfgs(cast(int)(2*S.length), p, &finalValue, &evaluate_maxent, &progress, &temp, &param);
	        foreach(j; 0..(2*S.length)) {
	        	weights[j] = p[j];
	        }
	        
	        debug {
	        	writeln("\n Penalty Method iteration: ", i, " - Weights: ", weights);
	        	writeln();
	        }
        	
        	v[] *= 2;
        	lambda[] *= 2;
        }
        debug {
        	writeln("LBFGS Result: ", ret);
        }	

        debug {
        	writeln("\nE: ", E,"\nQ: ",Q,"\nS: ",S,"\n learned p ",weights);
        }
        
        opt_value = finalValue;
                
     
        return map_p_onto_features(weights, S, model.getNumObFeatures(), lbfgs_use_ones, useHierDistr);
	}


	override double evaluate(double [] p, out double [] g, double step) {
    	
    	double returnval = 0;
    	
    	foreach(pi; p) {
    		returnval += pi * log(pi);
    		if (pi <= 0)
    			returnval = double.infinity;
    	}
    	
    	
    	foreach(j,Ej; E) {
    		double mult = 1.0;
    		foreach(i,Si; S) {
    			auto intersection = setIntersection(Si, Ej);
    			
    			if (equal(intersection, Si)) {
    				mult *= p[i];
    			}
    		}
    		mult -= Q[j];
    		mult = mult*mult;
    		mult *= v[j] / 2;
    		
    		returnval += mult;
    	}	
    	
    	foreach(i,Si; S) {
    		returnval += (lambda[i] / 2) * (p[i] + p[i + S.length] - 1)*(p[i] + p[i + S.length] - 1);
    	
    	}
    	
    	
    	g.length = 2*S.length;
    	g[] = 0;
    	

    	foreach(i,Si; S) {
    		if (p[i] <= 0) {
    			g[i] = - double.infinity;
    			continue;
    		}
    		g[i] = log(p[i]) + 1;
    		foreach(j,Ej; E) {
	    		double mult = v[j];
	    		
	    		foreach (k, Sk; S) {
	    			if (k != i) {
		    			auto intersection = setIntersection(Sk, Ej);
		    			
		    			if (equal(intersection, Sk)) {
		    				mult *= p[k];
		    			}
		    		} 
    			} 
	    		double mult2 = 1.0;
	    		foreach(l,Sl; S) {
	    			auto intersection2 = setIntersection(Sl, Ej);
	    			
	    			if (equal(intersection2, Sl)) {
	    				mult2 *= p[l];
	    			}
	    		}
	    		mult2 -= Q[j];
	    		mult *= mult2;
	    		
    			g[i] += mult;
    		}
    		g[i] += lambda[i] * (p[i] + p[i + S.length] - 1);
    	}
    	
    	
    	foreach(i,Si; S) {
    		if (p[i + S.length] <= 0 ) {
    			g[i + S.length] = -double.infinity;
    			continue;
    		}    	
    		g[i + S.length] = log(p[i + S.length]) + 1 + lambda[i] * (p[i] + p[i + S.length] - 1);
    	} 	
    	return returnval;
    	
	}

}