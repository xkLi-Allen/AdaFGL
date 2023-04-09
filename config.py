import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dirichlet_alpha', help='dirichlet distribution parameter', type=float,
                    default=0.5)

parser.add_argument('--gpu_id', help='', type=int,
                    default=0)

##

parser.add_argument('--threshold', help='', type=float,
                    default=0.9) #

parser.add_argument('--alpha', help='', type=float,
                    default=0.05)    

parser.add_argument('--beta', help='', type=float,
                    default=0.2)  

##
parser.add_argument('--seed', help='seed everything', type=int,
                    default=2023)

parser.add_argument('--partition', help='data simulation method', type=str,
                    default="Metis")

parser.add_argument('--data_name', help='dataset name', type=str,
                    default="PubMed")     

parser.add_argument('--num_clients', help='number of clients', type=int,
                    default=10)
        
parser.add_argument('--gmodel_name', help='global model name', type=str,
                    default="ChebNet")

parser.add_argument('--num_rounds', help='number of global model training rounds', type=int,
                    default=100)

parser.add_argument('--num_epochs', help='number of global model local training epochs', type=int,
                    default=3)


parser.add_argument('--lr', help='global model learning rate', type=float,
                    default=1e-2)

parser.add_argument('--weight_decay', help='global model weight decay', type=float,
                    default=5e-4)

parser.add_argument('--drop', help='global model drop out prob', type=float,
                    default=0.5)
            


parser.add_argument('--normalize_train', help='number of personal local model training times', type=int,
                    default=1)

parser.add_argument('--hidden_dim', help='multi layer model hidden units', type=int,
                    default=64)

parser.add_argument('--epochs', help='personalized propagation epoch', type=int,
                    default=200)



    
args = parser.parse_args()
print(args)

