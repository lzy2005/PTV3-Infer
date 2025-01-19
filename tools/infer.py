from pointcept.engines.infer import Inferer
from pointcept.utils.parser import argument_parser

def main():
	args = argument_parser().parse_args()
	cfg = args.__dict__
	inferer = Inferer(cfg)
	inferer.infer()

if __name__ == "__main__":
	main()
