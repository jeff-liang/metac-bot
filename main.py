import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal, Tuple, Optional

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


class CriticChallengerBot2025(ForecastBot):
    """
    Enhanced forecasting bot with a critic-challenger system.

    After the main forecaster makes an initial prediction, a critic LLM with higher temperature
    challenges the reasoning and prediction. They engage in a back-and-forth dialogue until
    they reach agreement or hit the maximum number of rounds.

    This adversarial approach helps:
    - Identify blind spots in reasoning
    - Challenge overconfident predictions
    - Consider alternative scenarios
    - Reach more robust consensus predictions
    """

    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # Critic-challenger parameters
    _max_dialogue_rounds = 7  # Maximum back-and-forth rounds

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Run research phase - unchanged from original"""
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif researcher == "asknews/deep-research/medium-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=2,
                    max_depth=4,
                )
            elif researcher == "asknews/deep-research/high-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=4,
                    max_depth=6,
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def _critic_challenger_dialogue(
            self,
            question_text: str,
            research: str,
            initial_reasoning: str,
            initial_prediction: str,
            question_type_instr: str,
            additional_context: str = ""
    ) -> Tuple[str, str]:
        """
        Facilitate a dialogue between the forecaster and critic until convergence.

        Returns:
            Tuple of (final_reasoning, final_prediction)
        """
        dialogue_history = [f"Initial Forecaster Reasoning: {initial_reasoning}"]
        current_reasoning = initial_reasoning
        current_prediction = initial_prediction

        for round_num in range(self._max_dialogue_rounds):
            logger.info(f"Critic-Challenger Round {round_num + 1}/{self._max_dialogue_rounds}")

            # Critic challenges the current prediction
            critic_prompt = clean_indents(
                f"""
                You are a critical analyst reviewing a forecast. Your job is to challenge the reasoning and identify potential flaws.

                Question: {question_text}

                Research available: {research}

                {additional_context}

                {"Previous dialogue:" + chr(10).join(dialogue_history)}

                Critically analyze this forecast by:
                1. Identifying potential biases or blind spots in the reasoning
                2. Suggesting alternative scenarios not fully considered
                3. Questioning the confidence level - is it too high or too low?
                4. Pointing out any logical inconsistencies
                5. Suggesting what additional factors should be weighted differently

                If you think the forecast is reasonable and well-justified, say "I AGREE with this forecast" and explain why.
                Otherwise, provide your critique and suggest what the prediction should be instead.

                End your response with either:
                - "I AGREE - Final prediction: [prediction]" if you agree
                - "I DISAGREE - Suggested prediction: [your prediction]" if you disagree
                """
            )

            critic_response = await self.get_llm("critic", "llm").invoke(critic_prompt)
            logger.info(f"Critic response: {critic_response[:500]}...")

            # Check if critic agrees
            if "I AGREE" in critic_response.upper():
                logger.info("Critic agrees with forecaster - consensus reached")
                final_reasoning = f"{current_reasoning}\n\nCritic's Agreement:\n{critic_response}"
                return final_reasoning, current_prediction

            # Forecaster responds to criticism
            forecaster_response_prompt = clean_indents(
                f"""
                You are the original forecaster. A critic has challenged your prediction.

                Question: {question_text}

                Research: {research}

                {additional_context}

                {"Previous dialogue:" + chr(10).join(dialogue_history)}

                Consider the critic's points carefully. You should:
                1. Acknowledge valid criticisms
                2. Defend aspects of your reasoning you still believe are correct
                3. Adjust your prediction if the critic made compelling points
                4. Provide updated reasoning incorporating the valuable insights

                Provide your updated reasoning and end with:
                "Updated prediction: [your final prediction for this round]"
                where the prediction is formatted according to these rules:
                {question_type_instr}
                """
            )

            forecaster_response = await self.get_llm("default", "llm").invoke(forecaster_response_prompt)
            logger.info(f"Forecaster response: {forecaster_response[:500]}...")

            # Extract updated prediction from forecaster's response
            if "Updated prediction:" in forecaster_response:
                prediction_start = forecaster_response.rfind("Updated prediction:")
                new_prediction = forecaster_response[prediction_start:].replace("Updated prediction:", "").strip()

                current_reasoning = forecaster_response
                current_prediction = new_prediction

            # Add to dialogue history
            dialogue_history.append(f"Round {round_num + 1} Critic: {critic_response}")
            dialogue_history.append(f"Round {round_num + 1} Forecaster: {forecaster_response}")

        logger.info(f"Max dialogue rounds reached. Final prediction: {current_prediction}")
        return current_reasoning, current_prediction

    async def _run_forecast_on_binary(
            self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # Initial forecast
        initial_prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ.Z%", 0.1-99.9
            """
        )

        initial_reasoning = await self.get_llm("default", "llm").invoke(initial_prompt)
        logger.info(f"Initial reasoning for URL {question.page_url}: {initial_reasoning}")

        # Extract initial prediction
        initial_binary_prediction: BinaryPrediction = await structure_output(
            initial_reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        initial_pred_str = f"{initial_binary_prediction.prediction_in_decimal * 100:.0f}%"

        # Run critic-challenger dialogue
        additional_context = f"""
        Background: {question.background_info}
        Resolution criteria: {question.resolution_criteria}
        Fine print: {question.fine_print}
        """

        final_reasoning, final_pred_str = await self._critic_challenger_dialogue(
            question.question_text,
            research,
            initial_reasoning,
            initial_pred_str,
            "ZZ.Z%, 0.1-99.9",
            additional_context
        )

        # Parse final prediction
        final_binary_prediction: BinaryPrediction = await structure_output(
            f"Probability: {final_pred_str}", BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, final_binary_prediction.prediction_in_decimal))

        logger.info(f"Final forecasted URL {question.page_url} with prediction: {decimal_pred}")
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=final_reasoning)

    async def _run_forecast_on_multiple_choice(
            self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        initial_prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )

        initial_reasoning = await self.get_llm("default", "llm").invoke(initial_prompt)
        logger.info(f"Initial reasoning for URL {question.page_url}: {initial_reasoning}")

        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )

        initial_predicted_options: PredictedOptionList = await structure_output(
            text_to_structure=initial_reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )

        # Format initial prediction for dialogue
        initial_pred_str = "\n".join(
            [f"{opt.option_name}: {opt.probability}" for opt in initial_predicted_options.predicted_options])

        # Run critic-challenger dialogue
        additional_context = f"""
        Options: {question.options}
        Background: {question.background_info}
        Resolution criteria: {question.resolution_criteria}
        Fine print: {question.fine_print}
        """

        final_reasoning, final_pred_str = await self._critic_challenger_dialogue(
            question.question_text,
            research,
            initial_reasoning,
            initial_pred_str,
            f"""your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N""",
            additional_context
        )

        # Parse final prediction
        final_predicted_options: PredictedOptionList = await structure_output(
            text_to_structure=final_pred_str,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )

        logger.info(f"Final forecasted URL {question.page_url} with prediction: {final_predicted_options}")
        return ReasonedPrediction(
            prediction_value=final_predicted_options, reasoning=final_reasoning
        )

    async def _run_forecast_on_numeric(
            self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )

        initial_prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )

        initial_reasoning = await self.get_llm("default", "llm").invoke(initial_prompt)
        logger.info(f"Initial reasoning for URL {question.page_url}: {initial_reasoning}")

        initial_percentile_list: list[Percentile] = await structure_output(
            initial_reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )

        # Format initial prediction for dialogue
        initial_pred_str = "\n".join([f"Percentile {p.percentile}: {p.value}" for p in initial_percentile_list])

        # Run critic-challenger dialogue
        additional_context = f"""
        Background: {question.background_info}
        Resolution criteria: {question.resolution_criteria}
        Fine print: {question.fine_print}
        Units: {question.unit_of_measure if question.unit_of_measure else "Not stated"}
        {lower_bound_message}
        {upper_bound_message}
        """

        final_reasoning, final_pred_str = await self._critic_challenger_dialogue(
            question.question_text,
            research,
            initial_reasoning,
            initial_pred_str,
            """Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX""",
            additional_context
        )

        # Parse final prediction
        final_percentile_list: list[Percentile] = await structure_output(
            final_pred_str, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(final_percentile_list, question)

        logger.info(f"Final forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=final_reasoning)

    def _create_upper_and_lower_bound_messages(
            self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the CriticChallengerBot forecasting system with adversarial dialogue"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    critic_bot = CriticChallengerBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/google/gemini-2.5-pro",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "critic": GeneralLlm(
                model="openrouter/x-ai/grok-4",  # Same model but different temperature
                temperature=0.8,  # Higher temperature for more creative challenges
                timeout=40,
                allowed_tries=2,
            ),
            "researcher": GeneralLlm(
                model="openrouter/perplexity/sonar-deep-research",
                timeout=40,
                allowed_tries=2,
            ),
        },
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            critic_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            critic_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        critic_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            critic_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        critic_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            critic_bot.forecast_questions(questions, return_exceptions=True)
        )
    critic_bot.log_report_summary(forecast_reports)