from . import dogvscat, mutag

class Finale:
    def __init__(self, args, data_loader, model):

        output_save = f"{args.output_root}/{args.config_name}"

        if args.dataset in dogvscat.__all__:
            dogvscat.__dict__[args.dataset](
                args.data_root,
                output_save, 
                args.section,
                args.data_format,
                args.image_size,
                args.maxv,
                data_loader,
                model
            )
        elif args.dataset in mutag.__all__:
            mutag.__dict__[args.dataset](
                args.data_root,
                output_save, 
                args.section,
                args.data_format,
                data_loader,
                model
            )            