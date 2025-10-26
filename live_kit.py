import asyncio
import logging
import base64
import cv2
import numpy as np

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    metrics,
    function_tool,
    RunContext,
)
from livekit.agents.llm import ImageContent
from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, vital_summary: str = None, video_frames: list = None) -> None:
        self.base_instructions = """You are Halo, a robotic baby monitoring system. You monitor infant vital activities and report abnormal readings to parents.
        You communicate in a mechanical, emotionless manner without any warmth, humor, or personal expressions.
        Your responses are clinical, precise, and data-driven. You state facts about vital signs, sleep patterns, movement, and other monitored parameters.
        You do not express concern, reassurance, or emotional support. You simply report status and abnormalities.
        Keep responses brief and technical. Use medical terminology when appropriate.
        You are a monitoring device, not a caregiver.
        
        CRITICAL: You must ONLY speak when there are abnormal vital signs or concerning observations. If all vitals are normal and there are no concerns, remain completely silent and do not generate any audio output.
        
        You will receive vital signs analysis summaries about heart rate and breathing count patterns from another monitoring model.
        These summaries contain inferences and analysis of time series data, not raw data.
        Use these analysis summaries to report any abnormalities or deviations from normal ranges.
        Reference specific time periods and measurements when reporting findings.
        
        You also have access to video frames from the monitoring camera. Analyze the visual data to assess infant positioning, movement patterns, and any visible concerns."""

        if vital_summary:
            instructions = f"{self.base_instructions}\n\nCurrent vital signs summary: {vital_summary}"
        else:
            instructions = self.base_instructions
        
        self.video_frames = video_frames or []
            
        super().__init__(instructions=instructions)

    @function_tool
    async def end_conversation(self, context: RunContext):
        logger.info("User requested to end conversation")
        raise Exception("User requested to end conversation")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext, vital_summary: str = None, video_frames: list = None):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        user_away_timeout=0,
        allow_interruptions=False,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    assistant = Assistant(vital_summary=vital_summary, video_frames=video_frames)
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(
            audio_enabled=False,
        )
    )

    if video_frames and len(video_frames) > 0:
        content_items = ["Here are video frames from the monitoring camera (captured over 5 seconds):"]
        frame_indices = np.linspace(0, len(video_frames) - 1, min(5, len(video_frames)), dtype=int)
        
        for idx in frame_indices:
            frame = video_frames[idx]
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            content_items.append(
                ImageContent(
                    image=f"data:image/jpeg;base64,{frame_base64}",
                    inference_width=512,
                    inference_height=512
                )
            )
        
        assistant.chat_ctx.add_message(
            role="user",
            content=content_items
        )

    await ctx.connect()
    
    await session.generate_reply(
        instructions="Analyze the vital signs summary and video frames if provided. If there are any abnormalities or concerns, report them briefly and clinically. If everything is normal, remain silent and generate no audio output.",
        allow_interruptions=False,
    )
    print("Session generated reply")
    


def call_live_kit(message, video_frames=None):
    async def entrypoint_with_sample(ctx: JobContext):
        return await entrypoint(ctx, message, video_frames)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint_with_sample, prewarm_fnc=prewarm))
