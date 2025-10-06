use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
    vec,
};

use burn_communication::{CommunicationChannel, Message, Protocol, ProtocolClient};
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, RwLock};

use crate::{NodeId, node::base::NodeState};

/// Handles the status of sync requests from other nodes
pub(crate) struct SyncService<P: Protocol> {
    /// Current node's state, shared with the thread that does aggregations
    node_state: Arc<RwLock<Option<NodeState>>>,
    /// The number of peers that have requested to sync with us since the last successful sync.
    syncing_peers: Mutex<Vec<NodeId>>,
    /// Notification on each incoming sync request
    sync_notif: Notify,

    _p: PhantomData<P>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SyncRequest(NodeId);

impl<P: Protocol> SyncService<P> {
    pub fn new(node_state: Arc<RwLock<Option<NodeState>>>) -> Self {
        Self {
            node_state,
            syncing_peers: Mutex::new(vec![]),
            sync_notif: Notify::new(),
            _p: PhantomData,
        }
    }

    fn add_syncing_peer(&self, peer: NodeId) {
        let mut syncing_peers = self.syncing_peers.lock().unwrap();
        syncing_peers.push(peer);
    }

    /// Sync with all peers.
    pub async fn sync(&self) {
        // we can't sync while we register
        let node_state = self.node_state.read().await;
        let node_state = node_state
            .as_ref()
            .expect("Trying to sync a node before having registered to the orchestrator");

        // this peer is syncing
        self.add_syncing_peer(node_state.node_id);
        for (id, addr) in &node_state.nodes {
            if *id == node_state.node_id {
                continue;
            }

            let mut connection = P::Client::connect(addr.clone(), "sync")
                .await
                .expect("Couldn't connect to peer for sync");
            let msg = SyncRequest(node_state.node_id);
            let sync_bytes = rmp_serde::to_vec(&msg).unwrap();
            connection
                .send(Message::new(sync_bytes.into()))
                .await
                .expect("Peer closed connection unexpectedly");
        }
        loop {
            {
                // compare currently synced peers with list of all nodes
                let mut syncing_peers = self.syncing_peers.lock().unwrap().to_vec();
                syncing_peers.sort();

                let mut all_node_ids = node_state.nodes.keys().cloned().collect::<Vec<_>>();
                all_node_ids.sort();

                if syncing_peers == all_node_ids {
                    // all nodes have synced
                    syncing_peers.clear();
                    return;
                }
            }
            // Wait for the next sync to come in
            self.sync_notif.notified().await
        }
    }

    pub async fn handle_sync_connection<C: CommunicationChannel>(&self, mut channel: C) {
        let msg = channel.recv().await.unwrap();
        let Some(msg) = msg else {
            return;
        };

        let msg = rmp_serde::from_slice::<SyncRequest>(&msg.data).unwrap();

        self.add_syncing_peer(msg.0);

        self.sync_notif.notify_waiters();
    }
}
