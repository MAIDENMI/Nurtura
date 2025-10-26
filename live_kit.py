import asyncio
import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    function_tool,
    RunContext,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, vital_summary: str = None) -> None:
        self.base_instructions = """You are Halo, a robotic baby monitoring system. You monitor infant vital activities and report abnormal readings to parents.
        You communicate in a mechanical, emotionless manner without any warmth, humor, or personal expressions.
        Your responses are clinical, precise, and data-driven. You state facts about vital signs, sleep patterns, movement, and other monitored parameters.
        You do not express concern, reassurance, or emotional support. You simply report status and abnormalities.
        Keep responses brief and technical. Use medical terminology when appropriate.
        You are a monitoring device, not a caregiver.
        
        You will receive vital signs analysis summaries about heart rate and breathing count patterns from another monitoring model.
        These summaries contain inferences and analysis of time series data, not raw data.
        Use these analysis summaries to report any abnormalities or deviations from normal ranges.
        Reference specific time periods and measurements when reporting findings."""
        

        if vital_summary:
            instructions = f"{self.base_instructions}\n\nCurrent vital signs summary: {vital_summary}"
        else:
            instructions = self.base_instructions
            
        super().__init__(instructions=instructions)

    @function_tool
    async def end_conversation(self, context: RunContext):
        logger.info("User requested to end conversation")
        raise Exception("User requested to end conversation")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext, vital_summary: str = None):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="assemblyai/universal-streaming:en",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Create assistant with vital summary if provided
    assistant = Assistant(vital_summary=vital_summary)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and introduce yourself as Halo, the baby monitoring system. Report the current vital signs status."
    )

    await ctx.connect()
    try:
        while True:
            await asyncio.sleep(1)
    except Exception:
        logger.info("Exception raised, ending monitoring session")
    finally:
        logger.info("Ending monitoring session")


if __name__ == "__main__":
    sample_vital_summary = "Heart rate: 120-140 bpm (normal range), Breathing: 30-40 breaths/min (normal), Sleep pattern: Regular REM cycles detected, Movement: Minimal activity during sleep phase, Alert: No abnormalities detected in last 2 hours"
    async def entrypoint_with_sample(ctx: JobContext):
        return await entrypoint(ctx, sample_vital_summary)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint_with_sample, prewarm_fnc=prewarm))
