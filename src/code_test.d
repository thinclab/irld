import std.stdio;
import std.math;
import std.random;
import mdp;
import sortingMDP;
import std.string;

void main() {

	string base_dir = "/home/katy/Desktop/Results_RI2RL/";
	string fname = base_dir~"noisyObsRobustSamplingMeirl_LBA_data.csv";
	File file1 = File(fname, "a"); 
	string str_arr_LBA = "0.98";
	str_arr_LBA = str_arr_LBA[1 .. (str_arr_LBA.length-1)];
	file1.writeln(str_arr_LBA);
	file1.close(); 

	// double[string] test = ["hi" : 1.5, "middle" : 2, "best": 3];
	// double* p;
	// p = ("hi" in test);
	// writeln((("hi" in test) !is null));
	// p = ("by" in test);
	// writeln((p !is null));
}