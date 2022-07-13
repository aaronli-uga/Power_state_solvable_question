using PowerModels
using Ipopt
using Distributions
using CSV
using DataFrames

baseMVA = 100
ntest = 10
max_ratio = 10
idx_mod_load = ["2", "3"]#index of loads to be modified
iffeas = zeros(ntest,1)
mod_ratio = rand(Uniform(-max_ratio,max_ratio),ntest,2)
iter_count = 1

for i in 1:ntest
  global iffeas
  global iter_count
  ratio = mod_ratio[i,:]
  network_data = PowerModels.parse_file("/Users/qili/Documents/PhD/UGA/Job&Intern/ANL_Internship/Workspace/code/matpower7.1/data/case39.m")
#----------------------change network data
  # ## branch
  # for idx_br in keys(network_data["branch"])
  #   network_data["branch"][idx_br]["br_r"] = network_data["branch"][idx_br]["br_r"] / baseR
  #   network_data["branch"][idx_br]["br_x"] = network_data["branch"][idx_br]["br_x"] / baseR
  # end
  # ## load
  # for idx_bus in keys(network_data["load"])
  #   network_data["load"][idx_bus]["pd"] = network_data["load"][idx_bus]["pd"] * kw2mw
  #   network_data["load"][idx_bus]["qd"] = network_data["load"][idx_bus]["qd"] * kw2mw
  # end
#----------------------------------------------------------------------
  # modify load
  network_data["load"][idx_mod_load[1]]["pd"] = network_data["load"][idx_mod_load[1]]["pd"] * (ratio[1]+1)
  network_data["load"][idx_mod_load[2]]["pd"] = network_data["load"][idx_mod_load[2]]["pd"] * (ratio[2]+1)
  ## solve OPF
  # result = run_opf(network_data, ACPPowerModel, Ipopt.Optimizer)
  # if result["termination_status"] == LOCALLY_INFEASIBLE
  #     iffeas[iter_count] = 0
  #   else
  #     iffeas[iter_count] = 1
  # end
  ## solve only PF
  result = run_pf(network_data, ACPPowerModel, Ipopt.Optimizer)
  if result["termination_status"] == LOCALLY_INFEASIBLE
      iffeas[iter_count] = 0 # label as infeasible
    else
      iffeas[iter_count] = 1 # label as feasible
  end
  iter_count = iter_count + 1
end
CSV.write("mod_ratio_new.csv",DataFrame(mod_ratio),writeheader=false)
CSV.write("iffeas_new.csv",DataFrame(iffeas),writeheader=false)
