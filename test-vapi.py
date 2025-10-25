from vapi import Vapi

if __name__ == "__main__":
    client = Vapi(
        token="56d4b3c7-65dc-45e2-a8b1-38444f96c8c5",
    )
    client.calls.create(
        assistant_id="f442ab7f-08ea-4ab6-a9fa-5b0ca5f3fda8",
        customers=[{"number": "+19037876024"}],
        phone_number_id="3ec65471-7581-4f32-8ccb-c3c17bd74070"
    )