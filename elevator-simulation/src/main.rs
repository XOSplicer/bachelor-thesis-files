
#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate rayon;
extern crate stats;
extern crate uom;

use std::iter;
use std::iter::IntoIterator;
use std::collections::{HashMap, VecDeque, HashSet};
use std::cmp;
use std::cmp::Ordering;
use rand::prelude::*;
use rand::distributions::{Distribution, Poisson};
use rayon::prelude::*;
use uom::si;
use uom::si::f32::*;
use stats::{OnlineStats, MinMax};

#[derive(Debug, Clone)]
struct BuildingParameters {
    floors: usize,
    interfloor_distance: Length,
    passengers_per_5_min: usize,
    cargo_per_5_min: usize,
}

impl Default for BuildingParameters {
    fn default() -> Self {
        BuildingParameters {
            floors: 8,
            interfloor_distance: Length::new::<si::length::meter>(3.5),
            passengers_per_5_min: 7,
            cargo_per_5_min: 2,
        }
    }
}

#[derive(Debug, Clone)]
struct LiftParameters {
    rated_weight: Mass,
    rated_passengers: usize,
    rated_speed: Velocity,
    door_opening_time: Time,
    door_closing_time: Time,
    flight_time_single_floor: Time,
}

impl Default for LiftParameters {
    fn default() -> Self {
        LiftParameters {
            rated_weight: Mass::new::<si::mass::kilogram>(5_000.0),
            rated_passengers: 65,
            rated_speed: Velocity::new::<si::velocity::meter_per_second>(1.6),
            door_opening_time: Time::new::<si::time::second>(1.5),
            door_closing_time: Time::new::<si::time::second>(1.5),
            flight_time_single_floor: Time::new::<si::time::second>(8.0),
        }
    }
}

#[derive(Debug, Clone)]
struct SimulationParameters {
    period: Time,
    slice: Time,
    simulations: usize,
}

impl Default for SimulationParameters {
    fn default() -> Self {
        SimulationParameters {
            period: Time::new::<si::time::hour>(1.0),
            slice: Time::new::<si::time::second>(0.1),
            simulations: 1000,
        }
    }
}

impl SimulationParameters {
    fn total_ticks(&self) -> usize {
        self.time_to_ticks(self.period)
    }

    fn time_to_ticks(&self, time: Time) -> usize {
        (time / self.slice).value.ceil() as usize
    }
}


#[derive(Debug)]
struct TrafficGenerator<'a>{
    sim: &'a SimulationParameters,
    building: &'a BuildingParameters,
    id_counter: usize,
}

impl<'a> TrafficGenerator<'a> {
    fn new(sim: &'a SimulationParameters, building: &'a BuildingParameters) -> Self {
        TrafficGenerator {
            sim,
            building,
            id_counter: 0,
        }
    }

    fn arrivals(&mut self) -> Vec<Vec<TrafficItem>> {
        (0..self.sim.total_ticks())
            .map(|i| self.arrivals_for_tick(i))
            .collect()
    }

    fn arrivals_for_tick(&mut self, tick: usize) -> Vec<TrafficItem> {
        let mut v = Vec::new();
        let mut rng = thread_rng();
        let pas = Self::random_amount_for_tick(
            self.building.passengers_per_5_min,
            self.sim.slice
        );
        //println!("pas: {}", pas);
        v.extend((0..pas).map(|_| {
            let mut p: TrafficItem =
                rng.choose(&PASSANGER_PROTOTYPES).unwrap().clone();
            p.id = self.id_counter;
            self.id_counter += 1;
            let from_to = Self::random_from_to(self.building.floors);
            p.from_floor = from_to.0;
            p.to_floor = from_to.1;
            p
        }));
        let carg = Self::random_amount_for_tick(
            self.building.cargo_per_5_min,
            self.sim.slice
        );
        v.extend((0..carg).map(|_| {
            let mut p: TrafficItem = rng.choose(&CARGO_PROTOTYPES).unwrap().clone();
            p.id = self.id_counter;
            self.id_counter += 1;
            let from_to = Self::random_from_to(self.building.floors);
            p.from_floor = from_to.0;
            p.to_floor = from_to.1;
            p
        }));
        v
    }

    fn random_from_to(floors: usize) -> (usize, usize) {
        assert!(floors > 0);
        let mut rng = thread_rng();
        let from: usize = rng.gen_range(0, floors);
        let others: Vec<usize> = (0..floors).filter(|&f| f != from).collect();
        let to = rng.choose(&others).unwrap();
        (from, *to)
    }

    fn random_amount_for_tick(per_5_min: usize, time_slice: Time) -> usize {
        let ticks_in_5_min =
            (Time::new::<si::time::minute>(5.0) / time_slice).value.ceil();
        let avg_per_tick = per_5_min as f32 / ticks_in_5_min;
        // println!("avg per tick: {}", avg_per_tick);
        let mut rng = thread_rng();
        let dist = Poisson::new(avg_per_tick.into());
        dist.sample(&mut rng) as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TrafficItemKind {
    Passenger,
    Cargo,
}

#[derive(Debug, Clone)]
struct TrafficItem {
    id: usize,
    kind: TrafficItemKind,
    width: Length,
    length: Length,
    height: Length,
    weight: Mass,
    from_floor: usize,
    to_floor: usize,
}

impl TrafficItem {
    fn transfer_time(&self) -> Time {
        match &self.kind {
            TrafficItemKind::Passenger => Time::new::<si::time::second>(2.0),
            TrafficItemKind::Cargo => Time::new::<si::time::second>(5.0),
        }
    }
}

lazy_static! {
    static ref PASSANGER_PROTOTYPES: Vec<TrafficItem> = vec![
        TrafficItem {
            id: 0,
            kind: TrafficItemKind::Passenger,
            width: Length::new::<si::length::centimeter>(50.0),
            length: Length::new::<si::length::centimeter>(25.0),
            height: Length::new::<si::length::centimeter>(160.0),
            weight: Mass::new::<si::mass::kilogram>(60.0),
            from_floor: 0,
            to_floor: 0,
        },
        TrafficItem {
            id: 0,
            kind: TrafficItemKind::Passenger,
            width: Length::new::<si::length::centimeter>(50.0),
            length: Length::new::<si::length::centimeter>(25.0),
            height: Length::new::<si::length::centimeter>(174.0),
            weight: Mass::new::<si::mass::kilogram>(65.0),
            from_floor: 0,
            to_floor: 0,
        },
        TrafficItem {
            id: 0,
            kind: TrafficItemKind::Passenger,
            width: Length::new::<si::length::centimeter>(50.0),
            length: Length::new::<si::length::centimeter>(25.0),
            height: Length::new::<si::length::centimeter>(180.0),
            weight: Mass::new::<si::mass::kilogram>(80.0),
            from_floor: 0,
            to_floor: 0,
        },
        TrafficItem {
            id: 0,
            kind: TrafficItemKind::Passenger,
            width: Length::new::<si::length::centimeter>(70.0),
            length: Length::new::<si::length::centimeter>(30.0),
            height: Length::new::<si::length::centimeter>(190.0),
            weight: Mass::new::<si::mass::kilogram>(110.0),
            from_floor: 0,
            to_floor: 0,
        }
    ];
}

lazy_static! {
    static ref CARGO_PROTOTYPES: Vec<TrafficItem> = vec![
        TrafficItem {
            id: 0,
            kind: TrafficItemKind::Cargo,
            width: Length::new::<si::length::meter>(2.5),
            length: Length::new::<si::length::meter>(1.5),
            height: Length::new::<si::length::meter>(1.5),
            weight: Mass::new::<si::mass::kilogram>(1000.0),
            from_floor: 0,
            to_floor: 0,
        },
        TrafficItem {
            id: 0,
            kind: TrafficItemKind::Cargo,
            width: Length::new::<si::length::meter>(2.5),
            length: Length::new::<si::length::meter>(1.5),
            height: Length::new::<si::length::meter>(1.5),
            weight: Mass::new::<si::mass::kilogram>(2000.0),
            from_floor: 0,
            to_floor: 0,
        },
        TrafficItem {
            id: 0,
            kind: TrafficItemKind::Cargo,
            width: Length::new::<si::length::meter>(2.5),
            length: Length::new::<si::length::meter>(1.5),
            height: Length::new::<si::length::meter>(1.5),
            weight: Mass::new::<si::mass::kilogram>(3000.0),
            from_floor: 0,
            to_floor: 0,
        },
        TrafficItem {
            id: 0,
            kind: TrafficItemKind::Cargo,
            width: Length::new::<si::length::meter>(2.5),
            length: Length::new::<si::length::meter>(1.5),
            height: Length::new::<si::length::meter>(1.5),
            weight: Mass::new::<si::mass::kilogram>(4000.0),
            from_floor: 0,
            to_floor: 0,
        },
        TrafficItem {
            id: 0,
            kind: TrafficItemKind::Cargo,
            width: Length::new::<si::length::meter>(2.5),
            length: Length::new::<si::length::meter>(1.5),
            height: Length::new::<si::length::meter>(1.5),
            weight: Mass::new::<si::mass::kilogram>(4900.0),
            from_floor: 0,
            to_floor: 0,
        }
    ];
}

#[derive(Debug)]
enum ControlAction {
    MoveUp,
    MoveDown,
    OpenDoors,
    CloseDoors,
    LoadTrafficItems(usize),
    UnloadTrafficItems,
    Nothing,
}

trait ControlStrategy {
    type State: Default;

    fn action(state: Self::State, sys: &ElevatorSystem) -> (Self::State, ControlAction);
    fn name() -> String;
}

struct SequentialControlStrategy;
#[derive(Default)]
struct SequentialControlState {
    car_call: Option<usize>,
    hall_calls: VecDeque<usize>,
}

impl ControlStrategy for SequentialControlStrategy {
    type State = SequentialControlState;
    fn name() -> String { "Sequential Control".into() }
    fn action(mut state: Self::State, sys: &ElevatorSystem) -> (Self::State, ControlAction) {

        // update hall calls
        for (f, q) in sys.floor_queues.iter().enumerate() {
            if !q.is_empty() && !state.hall_calls.contains(&f)  {
                state.hall_calls.push_back(f);
            }
        }

        // serve car call before hall calls
        if let Some(dest_f) = state.car_call {
            match dest_f.cmp(&sys.lift_floor) {
                Ordering::Equal => {
                    if sys.lift_doors_open {
                        // clear car call once passanger is delivered
                        state.car_call = None;
                        return (state, ControlAction::UnloadTrafficItems);
                    } else {
                        return (state, ControlAction::OpenDoors);
                    }
                },
                Ordering::Less => {
                    if sys.lift_doors_open {
                        return (state, ControlAction::CloseDoors);
                    } else {
                        return (state, ControlAction::MoveDown);
                    }
                },
                Ordering::Greater => {
                    if sys.lift_doors_open {
                        return (state, ControlAction::CloseDoors);
                    } else {
                        return (state, ControlAction::MoveUp);
                    }
                }
            }
        }

        // serve hall calls in order of arrival

        if let Some(dest_f) = state.hall_calls.front().cloned() {
            match dest_f.cmp(&sys.lift_floor) {
                Ordering::Equal => {
                    if sys.lift_doors_open {
                        if !sys.lift_traffic_items.is_empty() {
                            // clear hall call after loading and set car call
                            let _ = state.hall_calls.pop_front();
                            let car_call_f = sys.lift_traffic_items.values()
                                .flat_map(|v| v.iter())
                                .next()
                                // lets hope there was one to enter the lift
                                .unwrap()
                                .to_floor;
                            state.car_call = Some(car_call_f);
                            return (state, ControlAction::CloseDoors);
                        } else {
                            return (state, ControlAction::LoadTrafficItems(1));
                        }
                    } else {
                        return (state, ControlAction::OpenDoors);
                    }
                },
                Ordering::Less => {
                    if sys.lift_doors_open {
                        return (state, ControlAction::CloseDoors);
                    } else {
                        return (state, ControlAction::MoveDown);
                    }
                },
                Ordering::Greater => {
                    if sys.lift_doors_open {
                        return (state, ControlAction::CloseDoors);
                    } else {
                        return (state, ControlAction::MoveUp);
                    }
                }
            }
        }

        //unimplemented!()
        //TODO: implement sequential
        (state, ControlAction::Nothing)
    }
}

// non-directional collective
struct CollectiveControlStrategy;
#[derive(Default)]
struct CollectiveControlState {
    direction: Option<Direction>
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Up,
    Down
}

impl ControlStrategy for CollectiveControlStrategy {
    type State = CollectiveControlState;
    fn name() -> String { "Collective Control".into() }
    fn action(mut state: Self::State, sys: &ElevatorSystem) -> (Self::State, ControlAction) {

        let car_calls: HashSet<usize> = sys.lift_traffic_items.keys().cloned().collect();
        let hall_calls: HashSet<usize> = sys.floor_queues.iter()
            .enumerate()
            .filter(|(_floor, items)| !items.is_empty())
            .map(|(floor, _items)| floor)
            .collect();

        // if
        if car_calls.contains(&sys.lift_floor)  {
            if sys.lift_doors_open {
                return (state, ControlAction::UnloadTrafficItems);
            } else {
                return (state, ControlAction::OpenDoors);
            }
        }

        if hall_calls.contains(&sys.lift_floor) {
            if sys.lift_doors_open {
                //try to load all of them
                return (state, ControlAction::LoadTrafficItems(sys.floor_queues[sys.lift_floor].len()));
            } else {
                return (state, ControlAction::OpenDoors);
            }
        }

        if sys.lift_doors_open {
            return (state, ControlAction::CloseDoors);
        }

        // if travel direction unset go for next hall hall
        if state.direction.is_none() {
            let nearest = hall_calls.iter()
                .min_by_key(|&&c| (sys.lift_floor as i32 - c as i32).abs() as usize)
                .cloned();
            if let Some(nearest) = nearest {
                if nearest > sys.lift_floor {
                    state.direction = Some(Direction::Up);
                    return (state, ControlAction::MoveUp);
                } else if nearest < sys.lift_floor {
                    state.direction = Some(Direction::Down);
                    return (state, ControlAction::MoveDown);
                }
            }
        }

        // if there exists a hall call in the travel direction go for it
        if let Some(Direction::Up) = state.direction {
            if hall_calls.iter().any(|&f| f > sys.lift_floor) {
                return (state, ControlAction::MoveUp);
            }
        }

        if let Some(Direction::Down) = state.direction {
            if hall_calls.iter().any(|&f| f < sys.lift_floor) {
                return (state, ControlAction::MoveDown);
            }
        }

        // if there exists any other hall call go for it
        // therefore simply stop and do nothing
        // and wait for next turn
        state.direction = None;

        (state, ControlAction::Nothing)
    }
}

// basically collective control, but when a cargo item is inside,
// its sequential control to deliver for this item
struct AdaptedControlStrategy;
type AdaptedControlState = CollectiveControlState;

impl ControlStrategy for AdaptedControlStrategy {
    type State = AdaptedControlState;
    fn name() -> String { "Adaptive Control".into() }
    fn action(mut state: Self::State, sys: &ElevatorSystem) -> (Self::State, ControlAction) {
        let contains_cargo_destination: Option<usize> = sys.lift_traffic_items.values()
            .flat_map(|v| v.iter())
            .filter(|i| i.kind == TrafficItemKind::Cargo)
            .map(|i| i.to_floor)
            .next()
            .clone();
        if let Some(dest_floor) = contains_cargo_destination {
            if sys.lift_floor == dest_floor {
                if sys.lift_doors_open {
                    //reset direction when delivered
                    state.direction = None;
                    return (state, ControlAction::UnloadTrafficItems);
                } else {
                    return (state, ControlAction::OpenDoors);
                }
            }
            if sys.lift_floor < dest_floor {
                if sys.lift_doors_open {
                    return (state, ControlAction::CloseDoors);
                } else {
                    return (state, ControlAction::MoveUp);
                }
            }
            if sys.lift_floor > dest_floor {
                if sys.lift_doors_open {
                    return (state, ControlAction::CloseDoors);
                } else {
                    return (state, ControlAction::MoveDown);
                }
            }
        } else {
            // default to cc
            return CollectiveControlStrategy::action(state, sys);
        }

        return (state, ControlAction::Nothing)
    }
}

#[derive(Debug, Default, Clone)]
struct SimulationResults {
    passengers_in_traffic: usize,
    cargo_in_traffic: usize,
    stops: usize,
    passengers_delivered: usize,
    cargo_delivered: usize,
    waiting_ticks: HashMap<usize, usize>,
    ride_ticks: HashMap<usize, usize>,
}

#[derive(Debug)]
struct ElevatorSystem<'a> {
    building: &'a BuildingParameters,
    lift: &'a LiftParameters,
    sim: &'a SimulationParameters,
    floor_queues: Vec<VecDeque<TrafficItem>>,
    lift_floor: usize,
    lift_doors_open: bool,
    lift_traffic_items: HashMap<usize, Vec<TrafficItem>>,
    ticks: usize
}

impl<'a> ElevatorSystem<'a> {

    fn new(
        building: &'a BuildingParameters,
        lift: &'a LiftParameters,
        sim: &'a SimulationParameters,
    ) -> Self {
        ElevatorSystem {
            building,
            lift,
            sim,
            floor_queues: iter::repeat(VecDeque::new())
                .take(building.floors)
                .collect(),
            lift_floor: 0,
            lift_doors_open: false,
            lift_traffic_items: HashMap::new(),
            ticks: 0,
        }
    }

    fn move_up(&mut self) -> Result<Time, String> {
        if self.lift_doors_open {
            return Err("Lift open".into());
        }
        if self.lift_floor >= self.building.floors - 1 {
            return Err("Lift in highest floor".into());
        }
        self.lift_floor += 1;
        Ok(self.lift.flight_time_single_floor)
    }

    fn move_down(&mut self) -> Result<Time, String> {
        if self.lift_doors_open {
            return Err("Lift open".into());
        }
        if self.lift_floor == 0 {
            return Err("Lift in ground floor".into());
        }
        self.lift_floor -= 1;
        Ok(self.lift.flight_time_single_floor)
    }

    fn open_doors(&mut self) -> Result<Time, String> {
        if self.lift_doors_open {
            Err("Already open".into())
        } else {
            self.lift_doors_open = true;
            Ok(self.lift.door_opening_time)
        }
    }

    fn close_doors(&mut self) -> Result<Time, String> {
        if !self.lift_doors_open {
            Err("Already closed".into())
        } else {
            self.lift_doors_open = false;
            Ok(self.lift.door_closing_time)
        }
    }

    // return the ones that did not fit
    fn load_traffic_items(&mut self, t: Vec<TrafficItem>) -> Result<(Time, Vec<TrafficItem>), String> {
        if !self.lift_doors_open {
            return Err("Doors closed".into());
        }
        let mut t = t.into_iter();
        let mut transfer_times = Time::new::<si::time::second>(0.0);
        while let Some(n) = t.next() {
            if self.lift_traffic_items.values()
                .flat_map(|v| v.iter())
                .map(|i| i.weight)
                .sum::<Mass>() + n.weight > self.lift.rated_weight
                || self.lift_traffic_items.len() + 1 > self.lift.rated_passengers
            {
                break;
            }
            let s = self.lift_traffic_items.entry(n.to_floor).or_insert_with(Vec::new);
            transfer_times += n.transfer_time();
            s.push(n);
        }
        Ok((transfer_times, t.collect()))
    }

    fn unload_traffic_items_for_floor(&mut self, floor: usize) -> Result<(Time, Vec<TrafficItem>), String> {
        if !self.lift_doors_open {
            return Err("Doors closed".into());
        }
        let unloaded = self.lift_traffic_items
            .remove(&floor)
            .unwrap_or_else(Vec::new);
        let transfer_times = unloaded.iter()
            .map(|n| n.transfer_time())
            .sum::<Time>();
        Ok((transfer_times, unloaded))
    }
}

fn main() {
    let b = BuildingParameters::default();
    let l = LiftParameters::default();
    let s = SimulationParameters::default();
    println!("{:#?}\n{:#?}\n{:#?}\ntotal_ticks: {}\n", &b, &l, &s, s.total_ticks());

    let results =
        rayon::iter::repeatn((), s.simulations) // enable parallel
        //iter::repeat(()).take(s.simulations)
        .enumerate()
        .inspect(|(num, _)| println!("running simulation {}", num))
        .map(|_| TrafficGenerator::new(&s, &b).arrivals())
        .map(|t| (
            run_single_simulation::<SequentialControlStrategy>(&b, &l, &s, &t),
            run_single_simulation::<CollectiveControlStrategy>(&b, &l, &s, &t),
            run_single_simulation::<AdaptedControlStrategy>(&b, &l, &s, &t),
            ))
        .collect::<Vec<_>>();
    let invalid = results.iter()
        .filter(|&(a, b, c)| a.is_err() || b.is_err() || c.is_err())
        .count();
    let valid: Vec<_> = results.into_iter()
        .filter(|&(ref a, ref b, ref c)| a.is_ok() && b.is_ok() && c.is_ok())
        .map(|(a, b, c)| (a.unwrap(), b.unwrap(), c.unwrap()))
        .collect();
    let valid_a = valid.iter().map(|&(ref a, ref _b, ref _c)| a);
    let valid_b = valid.iter().map(|&(ref _a, ref b, ref _c)| b);
    let valid_c = valid.iter().map(|&(ref _a, ref _b, ref c)| c);

    println!("invalid results: {}", invalid);
    println!("{}", SequentialControlStrategy::name());
    print_stats(valid_a);
    println!("{}", CollectiveControlStrategy::name());
    print_stats(valid_b);
    println!("{}", AdaptedControlStrategy::name());
    print_stats(valid_c);

    println!("\ndone.");

}

fn print_stats<'a, I: Iterator<Item = &'a SimulationResults> + Clone>(it: I) {
    let pit = it.clone()
        .map(|i| i.passengers_in_traffic as f32);
    let s = pit.clone().collect::<OnlineStats>();
    let m = pit.clone().collect::<MinMax<_>>();
    println!("{}: min={} max={} avg={} stddev={}", "passengers_in_traffic", m.min().unwrap(), m.max().unwrap(), s.mean(), s.stddev());

    let cit = it.clone()
        .map(|i| i.cargo_in_traffic as f32);
    let s = cit.clone().collect::<OnlineStats>();
    let m = cit.clone().collect::<MinMax<_>>();
    println!("{}: min={} max={} avg={} stddev={}", "cargo_in_traffic", m.min().unwrap(), m.max().unwrap(), s.mean(), s.stddev());

    let stops = it.clone()
        .map(|i| i.stops as f32);
    let s = stops.clone().collect::<OnlineStats>();
    let m = stops.clone().collect::<MinMax<_>>();
    println!("{}: min={} max={} avg={} stddev={}", "stops", m.min().unwrap(), m.max().unwrap(), s.mean(), s.stddev());

    let pd = it.clone()
        .map(|i| i.passengers_delivered as f32);
    let s = pd.clone().collect::<OnlineStats>();
    let m = pd.clone().collect::<MinMax<_>>();
    println!("{}: min={} max={} avg={} stddev={}", "passengers_delivered", m.min().unwrap(), m.max().unwrap(), s.mean(), s.stddev());

    let cd = it.clone()
        .map(|i| i.cargo_delivered as f32);
    let s = cd.clone().collect::<OnlineStats>();
    let m = cd.clone().collect::<MinMax<_>>();
    println!("{}: min={} max={} avg={} stddev={}", "cargo_delivered", m.min().unwrap(), m.max().unwrap(), s.mean(), s.stddev());

    let wt = it.clone()
        .flat_map(|i| i.waiting_ticks.values())
        .cloned()
        .map(|i| i as f32);
    let s = wt.clone().collect::<OnlineStats>();
    let m = wt.clone().collect::<MinMax<_>>();
    println!("{}: min={} max={} avg={} stddev={}", "waiting_ticks", m.min().unwrap(), m.max().unwrap(), s.mean(), s.stddev());

     let rt = it.clone()
        .flat_map(|i| i.ride_ticks.values())
        .cloned()
        .map(|i| i as f32);
    let s = rt.clone().collect::<OnlineStats>();
    let m = rt.clone().collect::<MinMax<_>>();
    println!("{}: min={} max={} avg={} stddev={}", "ride_ticks", m.min().unwrap(), m.max().unwrap(), s.mean(), s.stddev());

}


fn run_single_simulation<S: ControlStrategy>(
    building: &BuildingParameters,
    lift: &LiftParameters,
    sim: &SimulationParameters,
    traffic: &Vec<Vec<TrafficItem>>,
) -> Result<SimulationResults, String> {
    println!("running simulation: {:?}", S::name());
    let mut system = ElevatorSystem::new(building, lift, sim);
    // let total: usize = traffic.iter().map(|t| t.len()).sum();
    // println!("total traffic items in simulation: {}", total);

    let mut state = S::State::default();
    let mut results = SimulationResults::default();

    results.passengers_in_traffic = traffic.iter()
        .flat_map(|t| t)
        .filter(|&t| t.kind == TrafficItemKind::Passenger)
        .count();
    results.cargo_in_traffic = traffic.iter()
        .flat_map(|t| t)
        .filter(|&t| t.kind == TrafficItemKind::Cargo)
        .count();

    // println!("total passengers in simulation: {}", &results.passengers_in_traffic);
    // println!("total cargo in simulation: {}", &results.cargo_in_traffic);

    while system.ticks < sim.total_ticks() {
        let last_ticks = system.ticks;
        let (next_state, action) = S::action(state, &system);
        state = next_state;
        match action {
            ControlAction::MoveUp => {
                let time = system.move_up()?;
                system.ticks += sim.time_to_ticks(time);
            },
            ControlAction::MoveDown => {
                let time = system.move_down()?;
                system.ticks += sim.time_to_ticks(time);
            },
            ControlAction::OpenDoors => {
                let time = system.open_doors()?;
                results.stops += 1;
                system.ticks += sim.time_to_ticks(time);
            },
            ControlAction::CloseDoors => {
                let time = system.close_doors()?;
                system.ticks += sim.time_to_ticks(time);
            },
            ControlAction::LoadTrafficItems(amount) => {
                let mut to_load = Vec::with_capacity(amount);
                for _ in 0..amount {
                    if let Some(i) = system.floor_queues[system.lift_floor]
                        .pop_front()
                    {
                        to_load.push(i);
                    }
                }
                let (time, rest) = system.load_traffic_items(to_load)?;
                for i in rest {
                    system.floor_queues[system.lift_floor]
                        .push_front(i);
                }
                system.ticks += sim.time_to_ticks(time);
            },
            ControlAction::UnloadTrafficItems => {
                let current_floor = system.lift_floor;
                let (time, items) = system.unload_traffic_items_for_floor(current_floor)?;
                results.passengers_delivered += items.iter()
                    .filter(|&i| i.kind == TrafficItemKind::Passenger)
                    .count();
                results.cargo_delivered += items.iter()
                    .filter(|&i| i.kind == TrafficItemKind::Cargo)
                    .count();
                system.ticks += sim.time_to_ticks(time);
            },
            ControlAction::Nothing => {
                system.ticks += 1;
            }
        }

        let next_tick = cmp::min(system.ticks, sim.total_ticks());
        let tick_diff = next_tick - last_ticks;

        // update waiting times
        for id in system.floor_queues.iter()
            .flat_map(|v| v.iter())
            .map(|i| i.id)
        {
            *results.waiting_ticks.entry(id).or_insert(0) += tick_diff;
        }

        // update ride times
        for id in system.lift_traffic_items.values()
            .flat_map(|v| v.iter())
            .map(|i| i.id)
        {
            *results.ride_ticks.entry(id).or_insert(0) += tick_diff;
        }

        // advance ticks and update floor queues with traffic items
        let new_traffic_items = traffic[last_ticks..next_tick]
            .iter()
            .flat_map(|v| v)
            .cloned();
        for i in new_traffic_items {
            system.floor_queues[i.from_floor].push_back(i)
        }

    }

    // remove ride times of passenegers that did not reach destination yet
    // and are in the lift
    for id in system.lift_traffic_items.values()
        .flat_map(|v| v.iter())
        .map(|i| i.id)
    {
        results.ride_ticks.remove(&id);
    }

    // remove waiting times of passenegrs still waiting
    for id in system.floor_queues.iter()
        .flat_map(|v| v.iter())
        .map(|i| i.id)
    {
        results.waiting_ticks.remove(&id);
    }

    Ok(results)
}