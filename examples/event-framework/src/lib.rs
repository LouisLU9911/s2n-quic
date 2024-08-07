// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/// Example of a print subscriber which can print all events or only
/// connection related events.
pub mod print_event {
    use s2n_quic::provider::event;
    use s2n_quic::provider::event::{ConnectionMeta};
    use s2n_quic::provider::event::events::{PacketLost, PacketSent, AckRangeReceived };

    #[derive(Debug, Clone)]
    pub struct MyPrintSubscriber {
        // prints all events, including connection events
        pub print_all_events: bool,
        // prints only connection related events
        pub print_connection_events: bool,
    }

    impl event::Subscriber for MyPrintSubscriber {
        type ConnectionContext = ();

        /// Initialize the Connection Context.
        fn create_connection_context(
            &mut self,
            meta: &ConnectionMeta,
            info: &event::ConnectionInfo,
        ) -> Self::ConnectionContext {
            println!("{:?} {:?}", meta, info);
        }

        /// This event fires for all events.
        // fn on_event<M: event::Meta + core::fmt::Debug, E: event::Event + core::fmt::Debug>(
        //     &mut self,
        //     meta: &M,
        //     event: &E,
        // ) {
        //     if self.print_all_events {
        //         println!("event: {:?} {:?}", meta, event);
        //     }
        // }

        fn on_packet_sent(
            &mut self,
            context: &mut Self::ConnectionContext,
            meta: &ConnectionMeta,
            event: &PacketSent,
        ) {
            println!("on_packet_sent {:?} {:?} {:?}", context, meta, event);
        }

        fn on_packet_lost(
            &mut self,
            context: &mut Self::ConnectionContext,
            meta: &ConnectionMeta,
            event: &PacketLost<'_>,
        ) {
            println!("on_packet_lost {:?} {:?} {:?}", context, meta, event);
        }

        fn on_ack_range_received(
            &mut self,
            context: &mut Self::ConnectionContext,
            meta: &ConnectionMeta,
            event: &AckRangeReceived<'_>,
        ) {
            println!("on_ack_range_received {:?} {:?} {:?}", context, meta, event);
        }

        // This event fires only for connection-level events. Excluded are events which
        // happen prior to connection creation, e.g. `on_version_information`,
        // `on_endpoint_datagram_drop`.
        // fn on_connection_event<E: event::Event + core::fmt::Debug>(
        //     &mut self,
        //     context: &mut Self::ConnectionContext,
        //     meta: &ConnectionMeta,
        //     event: &E,
        // ) {
        //     if self.print_connection_events {
        //         println!("connection_event: {:?} {:?} {:?}", context, meta, event);
        //     }
        // }
    }
}

/// Example of a query subscriber which can be used to store event information; which
/// can then be queried from the application.
pub mod query_event {
    use s2n_quic::provider::{
        event,
        event::{events, ConnectionMeta},
    };

    #[derive(Debug, Clone)]
    pub struct MyQueryContext {
        // Record how many data packets are received
        pub packet_sent_count: usize,
        // Flag to control the packet counter behavior
        pub count_non_data_packets: bool,
    }

    #[derive(Default)]
    pub struct MyQuerySubscriber;

    impl event::Subscriber for MyQuerySubscriber {
        type ConnectionContext = MyQueryContext;

        /// Initialize the Connection Context.
        fn create_connection_context(
            &mut self,
            _meta: &events::ConnectionMeta,
            _info: &events::ConnectionInfo,
        ) -> Self::ConnectionContext {
            MyQueryContext {
                packet_sent_count: 0,
                count_non_data_packets: true,
            }
        }

        /// This event fires for every outgoing packet that is transmitted.
        fn on_packet_sent(
            &mut self,
            context: &mut Self::ConnectionContext,
            _meta: &ConnectionMeta,
            event: &events::PacketSent,
        ) {
            match event.packet_header {
                events::PacketHeader::ZeroRtt { .. } | events::PacketHeader::OneRtt { .. } => {
                    context.packet_sent_count += 1;
                }
                _ => {
                    if context.count_non_data_packets {
                        context.packet_sent_count += 1;
                    }
                }
            }
        }
    }

    impl Drop for MyQueryContext {
        // Execute some operations on the context before the Connection is dropped.
        fn drop(&mut self) {
            println!("{:?}", self);
        }
    }
}
