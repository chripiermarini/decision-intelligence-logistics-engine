from data.input_data import InputData
from data.processing.demand_processor import DemandProcessor
from data.processing.lanes_processor import LanesProcessor
from data.processing.destinations_processor import DestinationsProcessor
from data.processing.origin_processor import OriginsProcessor


class DataProcessor:

    @staticmethod
    def process(input_data: InputData) -> InputData:
        return InputData(
            demand_history=DemandProcessor.process(input_data.demand_history),
            destinations=DestinationsProcessor.process(input_data.destinations),
            lanes=LanesProcessor.process(input_data.lanes),
            origins=OriginsProcessor.process(input_data.origins),
        )
