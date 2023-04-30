nvcc -rdc=true ^
main.cu TimingGPU.cu Utilities.cu RandomForest.cu DecisionTree.cu Data.cu ids_fs_problem.cu coevo_de.cu ^
-IC:\00_repos\ResearchForks\RandomForest-CPP\include
-o main